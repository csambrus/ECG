#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PipelineStep = str | tuple[str, dict[str, Any] | None]


def module_name_from_script(script: str) -> str:
    return script.removesuffix(".py")


def default_function_name_from_script(script: str) -> str:
    return module_name_from_script(script).split(".")[-1]


def pyrepr(value: Any) -> str:
    return repr(value)


def build_import_cell(pipeline: list[PipelineStep], function_overrides: dict[str, str] | None = None) -> str:
    function_overrides = function_overrides or {}
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    for step in pipeline:
        script = step if isinstance(step, str) else step[0]
        module = module_name_from_script(script)
        func = function_overrides.get(script, default_function_name_from_script(script))
        key = (module, func)
        if key not in seen:
            seen.add(key)
            lines.append(f"from {module} import {func}")

    return "\n".join(lines)


def build_calls_cell(pipeline: list[PipelineStep], function_overrides: dict[str, str] | None = None) -> str:
    function_overrides = function_overrides or {}
    lines: list[str] = []

    for step in pipeline:
        if isinstance(step, str):
            script = step
            kwargs = {}
        else:
            script, kwargs = step
            kwargs = kwargs or {}

        func = function_overrides.get(script, default_function_name_from_script(script))
        if kwargs:
            arg_txt = ", ".join(f"{k}={pyrepr(v)}" for k, v in kwargs.items())
            lines.append(f"print('=== {func} ===')\n{func}({arg_txt})")
        else:
            lines.append(f"print('=== {func} ===')\n{func}()")

    return "\n\n".join(lines)


def build_notebook_dict(
    pipeline_name: str,
    pipeline: list[PipelineStep],
    function_overrides: dict[str, str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    title = title or f"{pipeline_name} pipeline notebook"
    function_overrides = function_overrides or {}

    import_cell = build_import_cell(pipeline, function_overrides=function_overrides)
    calls_cell = build_calls_cell(pipeline, function_overrides=function_overrides)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n",
                "\n",
                "Auto-generated notebook skeleton from a kwargs-based pipeline.\n",
                "Edit the function override map if any module name and function name differ.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import sys\n",
                "\n",
                "PROJECT_ROOT = Path.cwd()\n",
                "if str(PROJECT_ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(PROJECT_ROOT))\n",
                "\n",
                "%matplotlib inline\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [import_cell + "\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [calls_cell + "\n"],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(
    pipeline_name: str,
    pipeline: list[PipelineStep],
    output_path: str | Path,
    function_overrides: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    output_path = Path(output_path)
    nb = build_notebook_dict(
        pipeline_name=pipeline_name,
        pipeline=pipeline,
        function_overrides=function_overrides,
        title=title,
    )
    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
