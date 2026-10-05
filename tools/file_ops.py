import difflib
import fnmatch
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from tools.base import registry


@registry.register(description="Reads file contents with optional line range pagination.")
def read_file(path: str, max_lines: int = 200, offset: int = 1) -> Dict[str, Any]:
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"Target is a directory, not a file: {path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)
    start_idx = max(0, offset - 1)
    end_idx = min(total_lines, start_idx + max_lines)
    selected_lines = all_lines[start_idx:end_idx]

    formatted_content = "".join([f"{i + 1}: {line}" for i, line in enumerate(selected_lines, start=start_idx)])

    return {
        "path": str(file_path),
        "total_lines": total_lines,
        "offset": offset,
        "lines_returned": len(selected_lines),
        "content": formatted_content,
    }


@registry.register(description="Creates a new file or overwrites an existing file if overwrite=True.")
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    file_path = Path(path).resolve()
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"File '{path}' already exists. Pass overwrite=True to replace it.")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Successfully wrote {len(content)} characters to {file_path.name}."


@registry.register(description="Replaces target_text with replacement_text inside a file.")
def edit_file(path: str, target_text: str, replacement_text: str) -> str:
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        original = f.read()

    if target_text not in original:
        raise ValueError(f"Target text was not found in file '{file_path.name}'.")

    count = original.count(target_text)
    updated = original.replace(target_text, replacement_text, 1)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated)

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{file_path.name}",
        tofile=f"b/{file_path.name}",
        n=2,
    )
    diff_str = "".join(diff)

    return f"Successfully updated '{file_path.name}' (occurrences found: {count}). Diff:\n{diff_str}"


@registry.register(description="Searches for files matching a filename pattern (e.g. '*.py', '*.json').")
def search_files(pattern: str, directory: str = ".", max_results: int = 30) -> List[Dict[str, Any]]:
    base = Path(directory).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    matches = []
    for root, dirs, files in os.walk(base):
        # Exclude hidden and virtual environment directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "node_modules", "__pycache__")]
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                full_path = Path(root) / filename
                rel_path = full_path.relative_to(base)
                matches.append({
                    "name": filename,
                    "relative_path": str(rel_path),
                    "size_bytes": full_path.stat().st_size,
                })
                if len(matches) >= max_results:
                    return matches
    return matches


@registry.register(description="Searches for text content inside files (similar to grep).")
def grep_in_files(query: str, directory: str = ".", file_pattern: str = "*", max_matches: int = 30) -> List[Dict[str, Any]]:
    base = Path(directory).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    results = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "node_modules", "__pycache__")]
        for filename in files:
            if fnmatch.fnmatch(filename, file_pattern):
                full_path = Path(root) / filename
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, start=1):
                            if query.lower() in line.lower():
                                results.append({
                                    "file": str(full_path.relative_to(base)),
                                    "line_number": line_num,
                                    "line_content": line.strip()[:160],
                                })
                                if len(results) >= max_matches:
                                    return results
                except Exception:
                    continue
    return results


@registry.register(description="Safely moves a file to the workspace trash folder instead of permanent deletion.")
def safe_delete_file(path: str) -> str:
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    trash_dir = target.parent / ".trash"
    trash_dir.mkdir(parents=True, exist_ok=True)

    dest = trash_dir / f"{target.name}"
    counter = 1
    while dest.exists():
        dest = trash_dir / f"{target.stem}_{counter}{target.suffix}"
        counter += 1

    shutil.move(str(target), str(dest))
    return f"Moved '{target.name}' to trash at '{dest}'."
