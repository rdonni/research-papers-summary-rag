from unittest.mock import MagicMock

import pytest

from instadeep_technical_test.summarize import (
    build_summarize_chain,
    extract_file_name_from_path,
    format_results,
)


def test_build_summarize_chain_invalid_type():
    model = MagicMock()
    chain_type = "invalid_type"

    with pytest.raises(ValueError, match="Not a valid chain type"):
        build_summarize_chain(model, chain_type)


def test_extract_file_name_from_path():
    path = "/path/to/file/sample.txt"
    file_name = extract_file_name_from_path(path)
    assert file_name == "sample"


def test_format_results():
    research_papers = {"Title1": "Summary1", "Title2": "Summary2"}
    formatted_dict = format_results(research_papers)
    assert "Title1" in formatted_dict
    assert "Summary1" in formatted_dict
    assert "Title2" in formatted_dict
    assert "Summary2" in formatted_dict
