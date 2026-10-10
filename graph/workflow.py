"""LangGraph workflow construction and execution (Phase 7).

This module wires the six existing agents into a LangGraph ``StateGraph`` and
runs it. It owns three things and nothing else: the edge structure, the
structural description used for debugging and tests, and the conversion of the
finished graph state into a typed :class:`~graph.state.WorkflowResult`.

Graph shape::

    START
      |
      v
    Data Analysis --+
      |             |
      v             |
    Preprocessing --+
      |             |
      v             |
    Model Selection --+        every required node ends the workflow when it
      |             |         fails: the router returns END instead of the
      v             |         next node, so no downstream agent is ever
    Training -------+         entered with incomplete state
      |             |
      v             |
    Evaluation -----+---> [ Should Optimize? ] --no----> Model Persistence --+
      |                        |                                           |
      |                       yes                                          |
      v                        v                                           |
                                                                             |
    Optimization --------------------------------------------------+          |
      |                                                             |          |
      v                                                             v          |
    Model Persistence ----------------------------------------------+----------+

Phase 9 made Model Persistence the terminal node. Both branches of the
optimization decision converge on it, so ``--no-optimize`` still leaves an
artifact behind, and the workflow always ends by writing files.
    Optimization <-------------+
      |
      v
     END

``LANGGRAPH_MAX_STEPS`` is passed to LangGraph as a hard ceiling, so even a
routing bug cannot produce an unbounded run.

The graph is compiled without a checkpointer. Checkpointing would add a
persistence dependency (and a database) for no benefit here: the run is
in-process and its result is already returned in full as a
:class:`WorkflowResult`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from graph.llm_nodes import LLM_NODE_FUNCTIONS
from graph.nodes import NODE_FUNCTIONS
from graph.routing import (
    build_workflow_result,
    derive_workflow_status,
    describe_retry_policy,
    describe_routing_rule,
    make_step_router,
    optimization_destinations,
    route_after_evaluation,
    route_after_llm_evaluation,
    route_after_llm_node,
    route_after_optimization,
    route_to_end,
)
from graph.state import (
    GraphNode,
    GraphState,
    LLMGraphNode,
    WorkflowResult,
    create_graph_state,
)
from llm.service import build_service
from utils import config
from utils.logger import setup_logger
from utils.state import AgentState

logger = setup_logger("graph.workflow")

#: The node order the workflow follows. Kept here as the single source of truth
#: for the edge structure and the structural description.
WORKFLOW_ORDER: tuple = (
    GraphNode.DATA_ANALYSIS,
    GraphNode.PREPROCESSING,
    GraphNode.MODEL_SELECTION,
    GraphNode.TRAINING,
    GraphNode.EVALUATION,
    GraphNode.OPTIMIZATION,
    GraphNode.MODEL_PERSISTENCE,
)

#: Phase 9: the node that has to run after the optimization decision, whether the
#: search ran or not. Kept as data so the edge structure and the test expectation
#: are derived from one place.
PERSISTENCE_NODE: GraphNode = GraphNode.MODEL_PERSISTENCE

#: Phase 8: which optional interpretation node is inserted after which
#: deterministic node. Data mapping only - the graph builder turns it into edges.
LLM_INSERT_AFTER: Dict[GraphNode, LLMGraphNode] = {
    GraphNode.DATA_ANALYSIS: LLMGraphNode.DATA_ANALYSIS,
    GraphNode.OPTIMIZATION: LLMGraphNode.OPTIMIZATION,
}


#: Phase 10: the tracking entry point, resolved lazily so that ``tools.mlflow_tools``
#: is not imported until a run actually needs it. Kept as a module attribute rather
#: than a direct import for two reasons: the tracking module is a heavy optional
#: dependency, and a test must be able to substitute it to prove that the graph is
#: unaffected by tracking.
TRACKER_ATTRIBUTE = "track_workflow_run"


def _resolve_tracker() -> Optional[Callable[..., Any]]:
    """Return the tracking entry point, or None when tracking is unavailable.

    Returns:
        The callable, or None when ``tools.mlflow_tools`` cannot be imported at
        all. A missing optional dependency is an ordinary state - the same one a
        developer sees who never installed mlflow - so it is reported as
        "no tracker" rather than raised.
    """
    override = globals().get(TRACKER_ATTRIBUTE)
    if override is not None:
        return override
    try:
        from tools import mlflow_tools  # noqa: PLC0415 - deliberately deferred
    except Exception as error:  # noqa: BLE001 - any import failure means no tracker
        logger.info(
            "%s tracking is unavailable (%s: %s); the run was not recorded.",
            config.MLFLOW_LOG_PREFIX,
            type(error).__name__,
            error,
        )
        return None
    return mlflow_tools.track_workflow_run


def _add_llm_step(builder: Any, llm_node: LLMGraphNode, following: GraphNode) -> None:
    """Insert one interpretation node into the chain ``previous -> llm -> next``.

    Args:
        builder: The LangGraph ``StateGraph`` being built.
        llm_node: The interpretation node to insert.
        following: The deterministic node that runs after it.

    Both edges are conditional, matching the Phase 7 convention: the chain can be
    ended at any step. The router is :func:`~graph.routing.route_after_llm_node`,
    which reads no LLM state at all.
    """
    builder.add_conditional_edges(
        llm_node.value,
        route_after_llm_node(following),
        {following.value: following.value, END: END},
    )


def build_workflow(max_steps: Optional[int] = None, enable_llm: bool = False) -> Any:
    """Build and compile the LangGraph workflow.

    Args:
        max_steps: Override for the hard step ceiling. Defaults to
            ``LANGGRAPH_MAX_STEPS``.
        enable_llm: Whether to include the optional Phase 8 interpretation nodes.
            False, the default, builds exactly the Phase 7 structure, which is what
            keeps every existing behaviour and test intact.

    Returns:
        The compiled LangGraph application, ready for ``invoke``.

    Raises:
        RuntimeError: If LangGraph refuses the step ceiling, rather than letting
            the workflow run unbounded.
    """
    limit = int(max_steps if max_steps is not None else config.LANGGRAPH_MAX_STEPS)

    builder = StateGraph(GraphState)

    for node, function in NODE_FUNCTIONS.items():
        builder.add_node(node.value, function)

    if enable_llm:
        for node, function in LLM_NODE_FUNCTIONS.items():
            builder.add_node(node.value, function)

    builder.add_edge(START, GraphNode.DATA_ANALYSIS.value)

    # Every required node is followed by a conditional edge so that a failure
    # ends the run explicitly. The optimization edge additionally applies the
    # optimization decision.
    for current, following in zip(WORKFLOW_ORDER, WORKFLOW_ORDER[1:]):
        if current is GraphNode.EVALUATION:
            continue
        if current is GraphNode.OPTIMIZATION:
            # Phase 9: optimization converges on Model Persistence instead of
            # ending the run. With the LLM layer enabled the edge is rewired to
            # enter the interpretation node first, so a disabled run keeps the
            # shape below exactly.
            llm_after_optimization = (
                LLM_INSERT_AFTER.get(current) if enable_llm else None
            )
            if llm_after_optimization is None:
                builder.add_conditional_edges(
                    current.value,
                    route_after_optimization,
                    {PERSISTENCE_NODE.value: PERSISTENCE_NODE.value, END: END},
                )
            else:
                builder.add_conditional_edges(
                    current.value,
                    make_step_router(llm_after_optimization),
                    {llm_after_optimization.value: llm_after_optimization.value, END: END},
                )
                builder.add_conditional_edges(
                    llm_after_optimization.value,
                    route_after_llm_node(PERSISTENCE_NODE),
                    {PERSISTENCE_NODE.value: PERSISTENCE_NODE.value, END: END},
                )
            continue
        # Phase 8: an optional interpretation node is inserted after Data Analysis
        # and after Optimization. When the layer is enabled the edge is rewired to
        # enter that node first, so a disabled run keeps the 12-edge Phase 7 shape
        # exactly.
        llm_node = LLM_INSERT_AFTER.get(current) if enable_llm else None
        if llm_node is None:
            builder.add_conditional_edges(
                current.value,
                make_step_router(following),
                {following.value: following.value, END: END},
            )
            continue

        builder.add_conditional_edges(
            current.value,
            make_step_router(llm_node),
            {llm_node.value: llm_node.value, END: END},
        )
        _add_llm_step(builder, llm_node, following)

    # Evaluation routes to the interpretation node first, then the decision is
    # applied. Recording the decision before the LLM step is what keeps the LLM
    # from being able to influence it: the router below reads only the stored
    # decision and the recorded metrics.
    if enable_llm:
        builder.add_conditional_edges(
            GraphNode.EVALUATION.value,
            make_step_router(LLMGraphNode.EVALUATION),
            {LLMGraphNode.EVALUATION.value: LLMGraphNode.EVALUATION.value, END: END},
        )
        builder.add_conditional_edges(
            LLMGraphNode.EVALUATION.value,
            route_after_llm_evaluation,
            optimization_destinations(),
        )
    else:
        builder.add_conditional_edges(
            GraphNode.EVALUATION.value,
            route_after_evaluation,
            optimization_destinations(),
        )

    # Model Persistence is the terminal node. The edge is conditional for the same
    # reason every other one is: a run that was asked to stop ends here explicitly
    # instead of silently finishing the write.
    builder.add_conditional_edges(
        PERSISTENCE_NODE.value, route_to_end, {END: END}
    )

    try:
        return builder.compile()
    except TypeError as error:
        # Older LangGraph releases have no recursion/step ceiling parameter.
        # Failing loudly is better than silently running without a loop guard.
        raise RuntimeError(
            "The installed LangGraph version does not accept a step ceiling, so "
            "the workflow cannot guarantee a bounded run. Install a version that "
            "supports it or pass max_steps=None explicitly."
        ) from error


def run_workflow(
    *,
    dataset_path: Optional[str] = None,
    target_column: Optional[str] = None,
    agent_state: Optional[AgentState] = None,
    enable_optimization: Optional[bool] = None,
    cross_validation: bool = True,
    n_trials: int = config.OPTUNA_N_TRIALS,
    max_retries: Optional[int] = None,
    stop_on_failure: bool = config.LANGGRAPH_STOP_ON_FAILURE,
    app: Optional[Any] = None,
    enable_llm: bool = False,
    llm_service: Optional[Any] = None,
    enable_persistence: Optional[bool] = None,
    persistence_output_dir: Optional[str] = None,
    enable_tracking: Optional[bool] = None,
) -> WorkflowResult:
    """Run the whole workflow and return its typed result.

    Args:
        dataset_path: CSV file for the Data Analysis node.
        target_column: Explicit target column, or None to let the agent detect it.
        agent_state: An existing state to continue from; a new one is created when
            None.
        enable_optimization: Whether the optimization node may run. None uses
            ``LANGGRAPH_ENABLE_OPTIMIZATION``.
        cross_validation: Whether the training node may use cross-validation.
        n_trials: Trials per model for the optimization node.
        max_retries: Retry budget per node. None uses ``LANGGRAPH_MAX_RETRIES``.
        stop_on_failure: Whether a failed required node ends the workflow.
        app: A pre-compiled graph, mainly for tests. Built when None.
        enable_llm: Whether the optional Phase 8 interpretation nodes run. False,
            the default, builds the Phase 7 graph unchanged.
        llm_service: The service the interpretation nodes call. A default service is
            built when None and the layer is enabled.
        enable_persistence: Whether the Phase 9 persistence node may write files.
            None uses ``PERSISTENCE_ENABLED``.
        persistence_output_dir: Directory the artifacts are written to, or None
            for the configured one.
        enable_tracking: Whether the Phase 10 layer records the run in the local
            MLflow store. None uses ``MLFLOW_ENABLED``; the final decision is
            still ANDed with the environment switch, so an explicit True cannot
            override ``MLFLOW_ENABLED=false``.

    Returns:
        The :class:`WorkflowResult` describing the finished run.
    """
    if enable_llm and llm_service is None:
        llm_service = build_service(True)
    if enable_optimization is None:
        enable_optimization = bool(config.LANGGRAPH_ENABLE_OPTIMIZATION)
    if max_retries is None:
        max_retries = int(config.LANGGRAPH_MAX_RETRIES)
    if enable_persistence is None:
        enable_persistence = bool(config.PERSISTENCE_ENABLED)
    if enable_tracking is None:
        enable_tracking = bool(config.MLFLOW_ENABLED)

    initial: GraphState = create_graph_state(
        agent_state=agent_state,
        dataset_path=dataset_path,
        target_column=target_column,
        enable_optimization=bool(enable_optimization),
        cross_validation=cross_validation,
        n_trials=n_trials,
        max_retries=int(max_retries),
        stop_on_failure=stop_on_failure,
        enable_llm=bool(enable_llm),
        enable_persistence=bool(enable_persistence),
        persistence_output_dir=persistence_output_dir,
    )
    if enable_llm:
        # The service travels in the state rather than in a module global, so the
        # interpretation nodes stay pure and a test can inject a stub.
        initial["llm_service"] = llm_service

    # A pre-compiled `app` wins, but only when it matches the requested structure:
    # building a graph without the LLM nodes and then running it with the layer
    # enabled would silently skip the interpretation steps.
    if app is not None:
        graph = app
    else:
        graph = build_workflow(enable_llm=bool(enable_llm))
    logger.info(
        "%s workflow -> STARTED (optimization=%s, retries=%d)",
        config.LANGGRAPH_LOG_PREFIX,
        "enabled" if enable_optimization else "disabled",
        int(max_retries),
    )
    final: GraphState = graph.invoke(initial)

    # No node knows the final outcome, so the status is derived here and written
    # back into the state. Without this the state would still read "running"
    # and the closing log line would contradict the report.
    final["workflow_status"] = derive_workflow_status(final).value

    logger.info(
        "%s workflow -> FINISHED status=%s",
        config.LANGGRAPH_LOG_PREFIX,
        final["workflow_status"],
    )
    result = build_workflow_result(final)

    # Phase 10 records the run *after* it has finished rather than as a graph
    # node. A node could not open a run before the first stage ran, and adding
    # one would change the graph shape that Phase 7 defined and that the
    # structural tests assert on. Recording here also means the tracking call
    # receives the same finished result the report is built from, so the two can
    # never disagree.
    #
    # Resolved through the module attribute rather than imported directly so a
    # test can substitute it, and guarded because a tracking failure must not
    # change the workflow's outcome.
    _record_tracking = _resolve_tracker()
    if _record_tracking is not None:
        try:
            _record_tracking(
                final,
                result,
                dataset_path=dataset_path,
                workflow_status=final["workflow_status"],
                enable_optimization=bool(enable_optimization),
                enable_llm=bool(enable_llm),
                enable_persistence=bool(enable_persistence),
                # Both switches have to agree. `enable_tracking` is the caller's
                # request (app.py's --no-mlflow sets it to False for one run) and
                # `config.MLFLOW_ENABLED` is the environment switch (MLFLOW_ENABLED
                # in .env.example, which turns tracking off for every run). Passing
                # only the first would let an explicit True silently override
                # MLFLOW_ENABLED=false and record a run the read side refuses to
                # list, so the run would exist and still be reported as missing.
                enabled=bool(enable_tracking) and bool(config.MLFLOW_ENABLED),
            )
        except Exception as error:  # noqa: BLE001 - tracking must never raise here
            # A last-resort guard. `track_workflow_run` already contains its own
            # failures, so reaching this means the tracking module itself is
            # broken rather than merely unavailable. It is still logged rather
            # than passed through: the caller's machine learning result is
            # already computed and must not be lost to a reporting bug.
            logger.error(
                "%s experiment tracking raised unexpectedly and the run was not "
                "recorded: %s: %s. The machine learning results are unaffected.",
                config.MLFLOW_LOG_PREFIX,
                type(error).__name__,
                error,
            )
    return result


# --------------------------------------------------------------------------- #
# Structural description (debugging and tests, no extra dependency)
# --------------------------------------------------------------------------- #


def describe_structure(app: Optional[Any] = None) -> str:
    """Return a deterministic textual description of the graph structure.

    LangGraph's own ``draw_ascii`` needs the optional ``grandalf`` package, and a
    drawing would be a visualisation rather than a debugging aid. This walks the
    compiled node and edge lists instead, so the description costs nothing and is
    stable enough to assert on in tests.

    Args:
        app: A compiled graph. One is built when None.

    Returns:
        A multi-line structural summary. It is intended for tests and debugging
        and is never written to the log.
    """
    graph = app if app is not None else build_workflow()
    drawing = graph.get_graph()

    nodes = [name for name in drawing.nodes]
    edges = [
        (edge.source, edge.target, bool(edge.conditional))
        for edge in drawing.edges
    ]

    width = max((len(name) for name in nodes), default=4)
    lines: List[str] = ["Workflow structure:"]
    lines.append(f"  nodes ({len(nodes)}): " + ", ".join(nodes))
    lines.append("  edges:")
    for source, target, conditional in edges:
        kind = "conditional" if conditional else "direct"
        label = "END" if target == "__end__" else target
        origin = "START" if source == "__start__" else source
        lines.append(f"    {origin:<{width}} -> {label:<{width}}  [{kind}]")
    return "\n".join(lines)


#: The base edges that :func:`expected_structure` removes when the optional LLM
#: layer is enabled. Each one is replaced by a chain through an interpretation
#: node, because the router that used to sit directly after the deterministic node
#: now sits after its interpretation instead. Phase 8 rewired two of these;
#: Phase 9 rewired two more, because both optimization-decision exits and the
#: Optimization -> Model Persistence edge are now followed by an interpretation.
REWIRED_BY_LLM: frozenset = frozenset(
    {
        ("data_analysis", "preprocessing", True),
        ("evaluation", "optimization", True),
        ("evaluation", "model_persistence", True),
        ("optimization", "model_persistence", True),
    }
)


def expected_structure(enable_llm: bool = False) -> List[tuple]:
    """Return the edge list the workflow is required to have.

    Written out rather than derived from :data:`WORKFLOW_ORDER`, so the test
    expectation stays an independent statement of the required shape. Every
    ``(source, target, conditional)`` triple the compiled graph reports must
    appear here, and nothing else may.

    With ``enable_llm=False`` (the default) the shape is 14 edges: one direct edge
    from START, the five forward edges between consecutive nodes up to Evaluation,
    the conditional edge that resolves the optimization decision, the edge from
    Optimization to Model Persistence, and one conditional edge to END from each of
    the five required nodes, from Optimization and from Model Persistence.

    Phase 9 replaced the two ``-> END`` exits of the decision edge with one exit to
    Model Persistence. The reason is visible in the list: a run with
    ``--no-optimize`` still has to write an artifact, so "skip the search" now
    means "go straight to persistence" rather than "finish with nothing on disk".

    With ``enable_llm=True`` the three interpretation nodes are inserted, giving 20
    edges. Four edges are replaced by chains through an interpretation node
    (``data_analysis -> preprocessing``, both exits of the optimization decision, and
    ``optimization -> model_persistence``) and the rest are untouched. Two
    properties are visible in the list: every interpretation node is followed by a
    *deterministic* node, and the optimization decision is taken by a router that
    follows ``llm_evaluation`` and reads only recorded state.
    """
    base: List[tuple] = [
        ("__start__", "data_analysis", False),
        ("data_analysis", "preprocessing", True),
        ("preprocessing", "model_selection", True),
        ("model_selection", "training", True),
        ("training", "evaluation", True),
        ("evaluation", "optimization", True),
        ("evaluation", "model_persistence", True),
        ("optimization", "model_persistence", True),
        ("optimization", "__end__", True),
        ("data_analysis", "__end__", True),
        ("preprocessing", "__end__", True),
        ("model_selection", "__end__", True),
        ("training", "__end__", True),
        ("evaluation", "__end__", True),
        ("model_persistence", "__end__", True),
    ]
    if not enable_llm:
        return base

    # With the layer enabled, four edges are replaced by chains through an
    # interpretation node, because the *following* router now sits after the
    # interpretation instead of directly after the deterministic node:
    # `data_analysis -> preprocessing`, both exits of the optimization decision
    # (`evaluation -> optimization` and `evaluation -> model_persistence`), and
    # `optimization -> model_persistence`. Every other edge is unchanged, which is
    # what keeps the LLM layer a rewiring rather than a second workflow.
    rewired = [edge for edge in base if edge not in REWIRED_BY_LLM]

    llm_edges: List[tuple] = [
        # Data Analysis is followed by its interpretation, then Preprocessing.
        ("data_analysis", "llm_data_analysis", True),
        ("llm_data_analysis", "preprocessing", True),
        ("llm_data_analysis", "__end__", True),
        # Evaluation records its decision, is interpreted, and only then routes. The
        # "no optimization" outcome now leads to Model Persistence, not to the end.
        ("evaluation", "llm_evaluation", True),
        ("llm_evaluation", "optimization", True),
        ("llm_evaluation", "model_persistence", True),
        ("llm_evaluation", "__end__", True),
        # Optimization is followed by the planner, which then persists the model. The
        # `optimization -> END` exit already exists in the base list and is shared
        # by both shapes, so it is not repeated here.
        ("optimization", "llm_optimization", True),
        ("llm_optimization", "model_persistence", True),
        ("llm_optimization", "__end__", True),
    ]
    return rewired + llm_edges


# --------------------------------------------------------------------------- #
# CLI report
# --------------------------------------------------------------------------- #


def _heading(number: int, title: str) -> List[str]:
    """Return a numbered section heading with a matching underline."""
    label = f"{number}. {title}"
    return ["", label, "-" * len(label)]


def format_workflow_report(result: WorkflowResult) -> str:
    """Build the human readable workflow report.

    Baseline and optimized results are not repeated in full here; the existing
    phase reports already print them. This section reports only what the graph
    itself decided and did, and it prints one line per fact so a multi-line
    dataclass repr can never end up in the output.

    Args:
        result: The finished workflow result.

    Returns:
        The formatted report.
    """
    separator = "=" * 78
    state = result.final_agent_state
    lines: List[str] = [separator, "WORKFLOW REPORT (LangGraph)", separator]

    lines += _heading(1, "Workflow status")
    lines.append(f"  Workflow status  : {result.workflow_status.value}")
    lines.append(f"  Stopped early    : {result.stopped_early}")
    if result.step_limit_reached:
        lines.append(f"  Step ceiling hit : {config.LANGGRAPH_MAX_STEPS}")
    lines.append(f"  Failed node      : {result.failed_node or 'none'}")
    lines.append(f"  AgentState status: {state.status}")
    lines.append(f"  Wall-clock time  : {state.get_execution_time():.4f} s")

    lines += _heading(2, "Nodes executed")
    if not result.node_records:
        lines.append("  (no node ran)")
    for record in result.node_records:
        detail = f"{record.attempts} attempt(s)"
        if record.retries:
            detail += f", {record.retries} retry/retries"
        if record.error_type:
            detail += f", {record.error_type}"
        lines.append(
            f"  {record.display_name:<16} {record.status.value:<10} "
            f"{record.duration:>8.4f} s  {detail}"
        )
    for node in GraphNode.ordered():
        if not result.record_for(node):
            lines.append(f"  {node.display_name:<16} not entered")

    lines += _heading(3, "Optimization decision")
    lines.append(f"  Decision         : {result.optimization_decision.value}")
    lines.append(f"  Outcome          : {result.optimization_outcome.value}")
    lines.extend(f"  {line}" for line in _wrap(result.optimization_decision.reason))

    lines += _heading(4, "Retry mechanism")
    lines.append(f"  Max retries/node : {config.LANGGRAPH_MAX_RETRIES}")
    lines.append(f"  Retries used     : {result.total_retries}")
    lines.append(f"  Attempts total   : {result.total_attempts}")
    lines.extend(f"  {line}" for line in _wrap(describe_retry_policy()))

    lines += _heading(5, "Errors")
    if not result.errors:
        lines.append("  none")
    for error in result.errors:
        lines.append(
            f"  [{error.node}] attempt {error.attempt} "
            f"{'(retryable)' if error.retryable else '(not retried)'}"
        )
        lines.extend(f"      {line}" for line in _wrap(error.message))

    lines += _heading(6, "Final result")
    baseline = state.best_baseline_model
    if baseline is not None:
        lines.append(
            f"  BEST BASELINE    : {baseline.model_name} "
            f"(hold-out F1 {baseline.f1_score:.4f}, accuracy {baseline.accuracy:.4f})"
        )
    elif state.evaluation_report is not None and state.evaluation_report.best_model:
        winner = state.evaluation_report.best_model
        lines.append(
            f"  BEST BASELINE    : {winner.best_baseline_model} "
            f"(hold-out F1 {winner.f1_score:.4f})"
        )
    else:
        lines.append("  BEST BASELINE    : none (evaluation produced no result)")

    if state.has_optimization_report:
        optimized = state.optimization_report.best_optimized_model
        if optimized is not None:
            lines.append(
                f"  BEST OPTIMIZED   : {optimized.best_optimized_model} "
                f"(hold-out F1 {optimized.holdout_f1_score:.4f})"
            )
        else:
            lines.append("  BEST OPTIMIZED   : none (no tuned model could be recommended)")
    elif state.optimization_status == "failed":
        lines.append(
            "  BEST OPTIMIZED   : none (optimization failed; the Phase 5 result "
            "above is preserved)"
        )
    else:
        lines.append("  BEST OPTIMIZED   : none (optimization was not run)")
    lines.append(f"  Optimization     : {state.optimization_status}")

    lines += _heading(7, "Local LLM layer (optional)")
    lines.extend(_llm_section(result))

    lines += _heading(8, "Model persistence")
    lines.extend(_persistence_section(result))

    lines += _heading(9, "Notes and limitations")
    lines.extend(_bullet(describe_routing_rule()))
    lines.extend(_bullet(config.LANGGRAPH_LIMITATION_NO_LLM))
    lines.extend(
        _bullet(
            config.LANGGRAPH_LIMITATION_STEP_LIMIT.format(
                max_steps=config.LANGGRAPH_MAX_STEPS
            )
        )
    )
    lines.extend(_bullet(config.LANGGRAPH_RECOMMENDATION_NO_LLM))

    lines.append(separator)
    return "\n".join(lines)


def _persistence_section(result: WorkflowResult) -> List[str]:
    """Render what the Phase 9 node wrote, or why it wrote nothing.

    The section is always present. A reader has to be able to tell "skipped", "not
    attempted" and "failed" apart, because only the last two mean the dashboard will
    find nothing.

    Args:
        result: The finished workflow result.

    Returns:
        The rendered lines.
    """
    state = result.final_agent_state
    lines: List[str] = []
    lines.append(f"  Persistence status: {state.persistence_status}")

    report = state.persistence_report
    if report is None:
        lines.extend(_bullet(config.DASHBOARD_NO_ARTIFACT))
        lines.extend(_bullet(config.PERSISTENCE_LIMITATION_NO_RETRAIN))
        return lines

    selection = report.selection
    lines.append(
        f"  Final model    : {selection.model_name} "
        f"(source: {selection.selection_source})"
    )
    lines.append(f"  Hold-out F1    : {_metric(report.selected_f1_score)}")
    lines.append(f"  Final fit      : {report.final_fit_strategy}")
    lines.append(
        f"  Artifacts      : {report.model_artifact}, "
        f"{report.metadata_artifact}, {report.results_artifact}"
    )
    numerical = len(list(state.preprocessing_numerical_features or []))
    categorical = len(list(state.preprocessing_categorical_features or []))
    lines.append(
        f"  Features       : {report.feature_count} "
        f"({numerical} numerical, {categorical} categorical), "
        f"{report.class_count} class(es)"
    )
    wrapped = _wrap(selection.selection_reason)
    if wrapped:
        lines.append(f"  Selection      : {wrapped[0].strip()}")
        lines.extend(f"                    {line.strip()}" for line in wrapped[1:])
    lines.extend(_bullet(config.PERSISTENCE_NOTE_NO_METRICS))
    lines.extend(_bullet(config.PERSISTENCE_LIMITATION_SINGLE_SPLIT))
    lines.extend(_bullet(config.PERSISTENCE_LIMITATION_NO_UPLOAD))
    return lines


def _metric(value: Optional[float]) -> str:
    """Format a metric for the report, showing '-' when it is missing."""
    return "-" if value is None else f"{float(value):.4f}"


def _llm_section(result: WorkflowResult) -> List[str]:
    """Render the Phase 8 layer, one line per fact.

    The section is always present so a reader can tell "disabled" apart from "not
    implemented", and it states what the layer may not do before it states anything
    it produced.

    Args:
        result: The finished workflow result.

    Returns:
        The rendered lines.
    """
    lines: List[str] = []
    if not result.llm_enabled:
        lines.extend(_bullet(config.LLM_DISABLED_NOTE))
        lines.extend(_bullet(config.LLM_AUTHORITY_RULE))
        return lines

    lines.append("  Layer enabled    : True")
    lines.append(f"  Model            : {config.OLLAMA_MODEL}")
    lines.append(f"  Statuses         : {result.llm_statuses or '{}'}")

    for entry in result.llm_results:
        lines.append(f"  [{entry.role.display_name}] {entry.status.value}")
        if entry.detail:
            lines.extend(f"      {line}" for line in _wrap(entry.detail))

        interpretation = entry.interpretation
        if interpretation is not None:
            if interpretation.summary:
                lines.extend(
                    f"      summary: {line}" for line in _wrap(interpretation.summary)
                )
            for label, items in (
                ("observed", interpretation.observations),
                ("interpretation", interpretation.interpretations),
                ("hypothesis", interpretation.hypotheses),
                ("recommendation", interpretation.recommendations),
            ):
                for item in items:
                    lines.extend(f"      {label}: {line}" for line in _wrap(item))
            lines.append(f"      confidence (model-reported): {interpretation.confidence.value}")

        plan = entry.plan
        if plan is not None:
            if plan.summary:
                lines.extend(f"      summary: {line}" for line in _wrap(plan.summary))
            if plan.candidate_models:
                lines.append(f"      candidates: {', '.join(plan.candidate_models)}")
            if plan.focus_areas:
                lines.append(f"      focus: {', '.join(plan.focus_areas)}")
            if plan.reason:
                lines.extend(f"      reason: {line}" for line in _wrap(plan.reason))
            if plan.rejected_models:
                lines.append(
                    f"      rejected models: {', '.join(plan.rejected_models)}"
                )
            if plan.rejected_focus_areas:
                lines.append(
                    "      rejected focus: "
                    + ", ".join(plan.rejected_focus_areas)
                )

        for note in entry.output.notes if entry.output is not None else ():
            lines.extend(f"      note: {line}" for line in _wrap(note))

    claims = result.llm_forbidden_claims
    if claims:
        lines.append(f"  Phrasing warnings: {len(claims)}")
        lines.extend(f"      {claim}" for claim in claims)

    lines.extend(_bullet(config.LLM_AUTHORITY_RULE))
    lines.extend(_bullet(config.LLM_LOCALITY_NOTE))
    lines.extend(_bullet(config.LLM_LIMITATION_GENERATED))
    lines.extend(_bullet(config.LLM_LIMITATION_LOCAL_MODEL))
    lines.extend(_bullet(config.LLM_LIMITATION_SINGLE_SPLIT))
    lines.extend(_bullet(config.LLM_LIMITATION_OPTUNA_AUTHORITY))
    lines.extend(_bullet(config.LLM_LIMITATION_NO_CLAIM_CHECK))
    lines.extend(_bullet(config.LLM_RECOMMENDATION_REVIEW))
    return lines


def _bullet(text: str, indent: str = "  ", width: int = 74) -> List[str]:
    """Render text as an indented bullet with hanging continuation lines.

    A word-wrapped bullet must not repeat its marker on every line, or a single
    wrapped paragraph reads as a list of separate points.

    Args:
        text: The text to render.
        indent: Leading whitespace for the bullet.
        width: Maximum line length, including the indent.

    Returns:
        The rendered bullet lines.
    """
    lines = _wrap(text, max(20, width - len(indent)))
    hanging = indent + "  "
    return [f"{indent}- {lines[0]}"] + [f"{hanging}{line}" for line in lines[1:]]


def _wrap(text: str, width: int = 74) -> List[str]:
    """Wrap text to a width without breaking words.

    Args:
        text: The text to wrap.
        width: Maximum line length.

    Returns:
        The wrapped lines, never empty.
    """
    words = str(text).split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
