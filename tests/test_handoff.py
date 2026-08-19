"""Handoff-note storage tests — always against tmp_path, never ~/.acode."""

from __future__ import annotations

from pathlib import Path

from acode.handoff import handoff_path, load_handoff, save_handoff


class TestHandoffStorage:
    def test_save_then_load_roundtrip(self, tmp_path: Path, workspace: Path) -> None:
        store = tmp_path / "store"
        path = save_handoff(workspace, "finish the parser\n", store)
        assert path is not None and path.is_file()
        loaded = load_handoff(workspace, store)
        assert loaded is not None
        date, note = loaded
        assert note == "finish the parser"
        assert len(date) == 10 and date.count("-") == 2  # YYYY-MM-DD

    def test_no_note_means_none(self, tmp_path: Path, workspace: Path) -> None:
        assert load_handoff(workspace, tmp_path / "empty-store") is None

    def test_blank_note_loads_as_none(self, tmp_path: Path, workspace: Path) -> None:
        store = tmp_path / "store"
        save_handoff(workspace, "   \n", store)
        assert load_handoff(workspace, store) is None

    def test_same_name_different_place_gets_a_different_note(self, tmp_path: Path) -> None:
        a = tmp_path / "one" / "repo"
        b = tmp_path / "two" / "repo"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        assert handoff_path(a, tmp_path) != handoff_path(b, tmp_path)

    def test_save_overwrites_the_previous_note(self, tmp_path: Path, workspace: Path) -> None:
        store = tmp_path / "store"
        save_handoff(workspace, "old note", store)
        save_handoff(workspace, "new note", store)
        loaded = load_handoff(workspace, store)
        assert loaded is not None and loaded[1] == "new note"

    def test_unwritable_store_reports_failure(self, tmp_path: Path, workspace: Path) -> None:
        blocked = tmp_path / "blocked"
        blocked.write_text("a file where the directory should go")
        assert save_handoff(workspace, "note", blocked) is None
