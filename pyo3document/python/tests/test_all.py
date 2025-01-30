import pytest
import pyo3document


def test_sum_as_string():
    assert pyo3document.sum_as_string(1, 1) == "2"
