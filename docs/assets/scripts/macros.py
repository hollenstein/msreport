import ast
import re
from pathlib import Path

API_DIR = Path("docs/api")
SRC_DIR = Path("msreport")
EXCLUDE_FILES = ["index.md"]


def define_env(env):
    @env.macro
    def api_module_overview_table():
        files = [f for f in API_DIR.glob("*.md") if f.name not in EXCLUDE_FILES]
        module_header = "Module" + "&nbsp;" * 20  # Hack to force a minimum col width
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
        table = "\n".join(table_entries)
        return table


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
