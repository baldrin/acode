"""Input-history tests. These mutate readline's process-global history, so
each test clears it before and after; they skip where readline (or its
clear_history) is unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from acode.display import _history_length, _truncate_history, setup_input_history

readline = pytest.importorskip("readline")

if not hasattr(readline, "clear_history"):  # pragma: no cover
    pytest.skip("readline backend cannot clear history", allow_module_level=True)


@pytest.fixture(autouse=True)
def clean_history() -> Iterator[None]:
    readline.clear_history()
    yield
    readline.clear_history()


class TestInputHistory:
    def test_history_roundtrips_through_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "history"
        saver = setup_input_history(path)
        assert saver is not None
        readline.add_history("first task")
        readline.add_history("second task")
        saver()
        assert path.is_file()

        readline.clear_history()
        setup_input_history(path)
        length = readline.get_current_history_length()
        assert length == 2
        assert readline.get_history_item(length) == "second task"

    def test_missing_history_file_is_fine(self, tmp_path: Path) -> None:
        assert setup_input_history(tmp_path / "never-written") is not None
        assert readline.get_current_history_length() == 0

    def test_saver_creates_the_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "history"
        saver = setup_input_history(path)
        assert saver is not None
        readline.add_history("task")
        saver()
        assert path.is_file()

    def test_truncate_forgets_prompt_answers(self) -> None:
        readline.add_history("real task")
        before = _history_length()
        readline.add_history("y")  # what input() records at an approval prompt
        _truncate_history(before)
        length = readline.get_current_history_length()
        assert length == 1
        assert readline.get_history_item(length) == "real task"

    def test_truncate_with_nothing_added_is_a_no_op(self) -> None:
        readline.add_history("real task")
        _truncate_history(_history_length())
        assert readline.get_current_history_length() == 1
