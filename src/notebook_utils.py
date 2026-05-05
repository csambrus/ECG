from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Callable
import json
import traceback


CHECKPOINT_DIR = Path(".notebook_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def checkpoint_path(section_name: str) -> Path:
    return CHECKPOINT_DIR / f"{str(section_name).lower().strip()}.json"


def _normalize_kwargs(kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    return kwargs or {}


def step_key(step_name: str, kwargs: dict[str, Any] | None = None) -> str:
    kwargs = _normalize_kwargs(kwargs)
    return f"{step_name}::{json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)}"


def step_to_key(step: str | tuple[str, dict[str, Any] | None]) -> str:
    if isinstance(step, str):
        return step_key(step_name=step, kwargs=None)

    step_name, kwargs = step
    return step_key(step_name=step_name, kwargs=kwargs)


def load_checkpoint(section_name: str) -> dict[str, Any]:
    path = checkpoint_path(section_name)

    if not path.exists():
        return {
            "section": section_name,
            "updated_at": None,
            "completed": [],
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("section", section_name)
    payload.setdefault("updated_at", None)
    payload.setdefault("completed", [])
    return payload


def load_completed(section_name: str) -> set[str]:
    payload = load_checkpoint(section_name)
    return set(payload.get("completed", []))


def save_checkpoint(section_name: str, completed: list[str] | set[str]) -> None:
    payload = {
        "section": section_name,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed": sorted(set(completed)),
    }

    checkpoint_path(section_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_completed(section_name: str, completed: set[str]) -> None:
    save_checkpoint(section_name, completed)


def should_run(
    section_name: str,
    step_name: str,
    kwargs: dict[str, Any] | None = None,
    force: bool = False,
) -> bool:
    if force:
        return True

    completed = load_completed(section_name)
    return step_key(step_name, kwargs) not in completed


def mark_done(
    section_name: str,
    step_name: str,
    kwargs: dict[str, Any] | None = None,
) -> None:
    completed = load_completed(section_name)
    completed.add(step_key(step_name, kwargs))
    save_completed(section_name, completed)


def unmark_done(
    section_name: str,
    step_name: str,
    kwargs: dict[str, Any] | None = None,
) -> None:
    completed = load_completed(section_name)
    completed.discard(step_key(step_name, kwargs))
    save_completed(section_name, completed)


def reset_section(section_name: str) -> None:
    path = checkpoint_path(section_name)

    if path.exists():
        path.unlink()
        print(f"Reset checkpoint: {path}")
    else:
        print(f"No checkpoint for: {section_name}")


def list_completed(section_name: str) -> list[str]:
    return sorted(load_completed(section_name))


def run_step_if_needed(
    section_name: str,
    func: Callable[..., Any],
    kwargs: dict[str, Any] | None = None,
    step_name: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> Any:
    kwargs = kwargs or {}

    if step_name is None:
        step_name = func.__name__

    key = step_key(step_name, kwargs)

    if not should_run(section_name, step_name, kwargs=kwargs, force=force):
        if verbose:
            print(f"[SKIP] {key}")
        return None

    if verbose:
        print(f"[RUN ] {key}")

    result = func(**kwargs)
    mark_done(section_name, step_name, kwargs=kwargs)

    if verbose:
        print(f"[DONE] {key}")

    return result


def run_section(
    section_name: str,
    pipeline: list[str | tuple[str, dict[str, Any] | None]],
    resolver: Callable[[str, dict[str, Any] | None], Any],
    resume: bool = True,
    force: bool = False,
    verbose: bool = True,
) -> None:
    """
    pipeline elemek:
      - "step_name"
      - ("step_name", {"arg": 1})

    resolver hívása:
      resolver(step_name, kwargs)
    """
    existing_completed = load_completed(section_name) if resume else set()
    completed = set(existing_completed)

    if not resume:
        save_completed(section_name, set())

    total = len(pipeline)

    for idx, step in enumerate(pipeline, start=1):
        if isinstance(step, str):
            step_name = step
            kwargs = None
        else:
            step_name, kwargs = step

        kwargs = _normalize_kwargs(kwargs)
        key = step_key(step_name, kwargs)

        if key in completed and not force:
            if verbose:
                print(f"[SKIP {idx}/{total}] {key}")
            continue

        try:
            if verbose:
                print(f"[RUN  {idx}/{total}] {key}")

            resolver(step_name, kwargs)

            completed.add(key)
            save_completed(section_name, completed)

            if verbose:
                print(f"[DONE {idx}/{total}] {key}")

        except Exception:
            print(f"[FAIL {idx}/{total}] {key}")
            traceback.print_exc()
            save_completed(section_name, completed)
            raise