"""Generate a virtual 'index.md' API overview page for documentation."""

import ast
import re
from pathlib import Path

import mkdocs_gen_files

DOCS_DIR = Path("docs")
API_DIR = Path("docs/api")
SRC_DIR = Path("msreport")
OUTPUT_FILE = "index.md"
EXCLUDE_FILES = [OUTPUT_FILE]
PAGE_CONTENT = "# API Overview"


def main():
    files = [f for f in API_DIR.glob("*.md") if f.name not in EXCLUDE_FILES]
    module_header = "Module" + "&nbsp;" * 20  # Workaround to force a minimum col width
    table_entries = [f"| {module_header} | Description |", "|---|---|"]
    for filename in sorted(files):
        module_name = filename.stem
        module_py_file = SRC_DIR / f"{module_name}.py"
        module_init_py_file = SRC_DIR / module_name / "__init__.py"

        if module_py_file.exists():
            docstring = _get_module_docstring(module_py_file)
        elif module_init_py_file.exists():
            docstring = _get_module_docstring(module_init_py_file)
        else:
            docstring = ""
        docstring_summary = _get_summary_from_docstring(docstring)
        table_entries.append(
            f"| [`{module_name}`]({filename.name}) | {docstring_summary} |"
        )

    api_rel_dir = API_DIR.relative_to(DOCS_DIR)  # Required for mkdocs-gen-files
    with mkdocs_gen_files.open(api_rel_dir / OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"{PAGE_CONTENT}\n\n")
        f.write("\n".join(table_entries))
        # print full path of f:
        print(f"Generated API overview at: {f}")


def _get_module_docstring(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        node = ast.parse(f.read())
    docstring = ast.get_docstring(node) or ""
    return docstring


def _get_summary_from_docstring(docstring):
    if not docstring:
        return ""
    # Split on one or more blank lines (which may contain spaces)
    parts = re.split(r"\n\s*\n", docstring.strip(), maxsplit=1)
    summary = parts[0].replace("\n", " ").strip()
    return summary


main()
