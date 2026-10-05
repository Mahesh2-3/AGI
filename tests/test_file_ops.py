import os
from pathlib import Path
from tools.base import registry
import tools.file_ops


def test_file_operations_lifecycle(tmp_path):
    test_file = tmp_path / "test_doc.txt"

    # Write file
    write_res = registry.execute("write_file", {"path": str(test_file), "content": "Hello Jarvis\nLine two\nLine three"})
    assert write_res.success is True

    # Read file
    read_res = registry.execute("read_file", {"path": str(test_file), "max_lines": 10})
    assert read_res.success is True
    assert "Hello Jarvis" in read_res.output["content"]
    assert read_res.output["total_lines"] == 3

    # Edit file
    edit_res = registry.execute("edit_file", {
        "path": str(test_file),
        "target_text": "Hello Jarvis",
        "replacement_text": "Greetings Sir",
    })
    assert edit_res.success is True
    assert "Diff:" in edit_res.output

    # Grep in file
    grep_res = registry.execute("grep_in_files", {
        "query": "Greetings",
        "directory": str(tmp_path),
    })
    assert grep_res.success is True
    assert len(grep_res.output) == 1
    assert "Greetings Sir" in grep_res.output[0]["line_content"]

    # Safe delete (move to trash)
    del_res = registry.execute("safe_delete_file", {"path": str(test_file)})
    assert del_res.success is True
    assert not test_file.exists()
    assert (tmp_path / ".trash" / "test_doc.txt").exists()


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        test_file_operations_lifecycle(Path(tmp))
    print("✅ All file operations tests passed successfully!")
