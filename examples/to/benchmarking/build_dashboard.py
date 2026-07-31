"""Build self-contained benchmark dashboards from aggregated results."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig


def _validate_media(output_dir: Path, summary: dict[str, Any]) -> None:
    missing = []
    for row in [*summary["baseline"], *summary["scaling"]]:
        for field in ("video", "poster"):
            path = output_dir / str(row[field])
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        raise ValueError("Missing dashboard media:\n" + "\n".join(missing))


def _render_template(
    template: str, summary: dict[str, Any], artifact_prefix: str
) -> str:
    payload = json.dumps(summary, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("</", "<\\/")
    marker = "__BENCHMARK_DATA__"
    prefix_marker = "__ARTIFACT_PREFIX__"
    if template.count(marker) != 1 or template.count(prefix_marker) != 7:
        raise ValueError("Dashboard template must contain one data marker")
    return template.replace(marker, payload).replace(
        prefix_marker, artifact_prefix
    )


def _prefixed_summary(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    adjusted = deepcopy(summary)
    for row in [*adjusted["baseline"], *adjusted["scaling"]]:
        row["video"] = prefix + str(row["video"])
        row["poster"] = prefix + str(row["poster"])
    return adjusted


def _run(cfg: DictConfig) -> None:
    output_dir = Path(str(cfg.output_dir)).resolve()
    summary = json.loads((output_dir / "summary.json").read_text())
    _validate_media(output_dir, summary)
    source_dir = Path(__file__).resolve().parent
    template = (source_dir / "dashboard.html").read_text(encoding="utf-8")
    (output_dir / "index.html").write_text(
        _render_template(template, summary, ""), encoding="utf-8"
    )

    dashboard_root = Path(str(cfg.dashboard_root)).resolve()
    dashboard_root.parent.mkdir(parents=True, exist_ok=True)
    relative_output = output_dir.relative_to(dashboard_root.parent).as_posix()
    root_summary = _prefixed_summary(summary, f"{relative_output}/")
    dashboard_root.write_text(
        _render_template(template, root_summary, f"{relative_output}/"),
        encoding="utf-8",
    )


@hydra.main(version_base=None, config_path=".", config_name="grid")
def main(cfg: DictConfig) -> None:
    """Build the run-local and repository-level HTML dashboards."""
    _run(cfg)


if __name__ == "__main__":
    main()
