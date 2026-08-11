"""Parsing of the model's <think> output format."""

from __future__ import annotations

from app.inference import split_reasoning


def test_splits_a_complete_think_block() -> None:
    answer, reasoning = split_reasoning(
        "<think>The pedestrian is entering the crosswalk.</think>No, it is not safe."
    )
    assert answer == "No, it is not safe."
    assert reasoning == "The pedestrian is entering the crosswalk."


def test_handles_multiline_reasoning() -> None:
    raw = "<think>\nStep one.\nStep two.\n</think>\n\nThe answer is 42.\n"
    answer, reasoning = split_reasoning(raw)
    assert answer == "The answer is 42."
    assert reasoning == "Step one.\nStep two."


def test_unterminated_block_yields_reasoning_and_no_answer() -> None:
    """What truncation looks like: the token cap hit mid-reasoning.

    Returning the partial reasoning as the answer would be worse than returning
    nothing, because it reads like a real answer.
    """
    answer, reasoning = split_reasoning("<think>I am still thinking about whether the")
    assert answer == ""
    assert reasoning == "I am still thinking about whether the"


def test_output_without_a_think_block_is_all_answer() -> None:
    answer, reasoning = split_reasoning("  Yes, proceed.  ")
    assert answer == "Yes, proceed."
    assert reasoning == ""


def test_angle_brackets_in_the_answer_are_not_mistaken_for_tags() -> None:
    answer, reasoning = split_reasoning("<think>math</think>Use a < b < c ordering.")
    assert answer == "Use a < b < c ordering."
    assert reasoning == "math"
