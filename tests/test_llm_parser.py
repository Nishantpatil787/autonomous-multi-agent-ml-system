"""Tests for llm/parser.py — JSON block extraction, malformed recovery, and schema verification."""

import pytest

from llm.parser import extract_json_block, parse_and_validate_llm_json


class TestJSONExtraction:
    def test_extract_from_clean_json(self):
        text = '{"summary": "dataset analysis", "confidence": "high"}'
        assert extract_json_block(text) == text

    def test_extract_from_markdown_fences(self):
        text = 'Here is the response:\n```json\n{"summary": "fence test"}\n```\nHope it helps.'
        extracted = extract_json_block(text)
        assert extracted == '{"summary": "fence test"}'

    def test_extract_from_surrounding_prose(self):
        text = 'Prefix narrative {"summary": "nested"} suffix narrative'
        extracted = extract_json_block(text)
        assert extracted == '{"summary": "nested"}'

    def test_extract_from_empty_string(self):
        assert extract_json_block("") is None
        assert extract_json_block(None) is None


class TestJSONValidation:
    def test_parse_valid_json_with_required_keys(self):
        raw = '{"summary": "Valid summary", "observations": ["fact 1"]}'
        valid, data, err = parse_and_validate_llm_json(raw, required_keys=["summary", "observations"])
        assert valid is True
        assert data["summary"] == "Valid summary"
        assert err is None

    def test_parse_missing_required_keys(self):
        raw = '{"summary": "Only summary"}'
        valid, data, err = parse_and_validate_llm_json(raw, required_keys=["summary", "observations"])
        assert valid is False
        assert "Missing required JSON fields" in err

    def test_parse_malformed_json_returns_error(self):
        raw = '{"summary": broken json'
        valid, data, err = parse_and_validate_llm_json(raw)
        assert valid is False
        assert "JSON decoding error" in err

    def test_parse_empty_content_returns_error(self):
        valid, data, err = parse_and_validate_llm_json("")
        assert valid is False
        assert "Empty response" in err

    def test_parse_non_dict_json_returns_error(self):
        raw = '["item1", "item2"]'
        valid, data, err = parse_and_validate_llm_json(raw)
        assert valid is False
        assert "Expected JSON object" in err
