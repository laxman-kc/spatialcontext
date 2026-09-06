"""Render the current memory comparisons from verified numerical snapshots."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ecqa-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "overview"
INK, MUTED, TEAL = "#17202a", "#59636e", "#08786d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_metrics(name: str) -> tuple[dict, dict]:
    path = ROOT / "results" / name / "metrics.json"
    return json.loads(path.read_text()), {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def count(value: dict) -> dict:
    total = value.get("count", value.get("total"))
    correct = value["correct"]
    if (not isinstance(total, int) or not isinstance(correct, int)
            or not 0 <= correct <= total or total <= 0
            or not math.isclose(value["accuracy"], correct / total, abs_tol=1e-12)):
        raise ValueError("Saved count and accuracy disagree")
    return {"correct": correct, "total": total, "accuracy": correct / total}


def main() -> None:
    memory, source_memory = read_metrics("memory")
    reader, source_reader = read_metrics("memory-reader")
    val = memory["splits"]["val"]["conditions"]
    regression = memory["splits"]["test"]["conditions"]
    tuned = reader["splits"]["val"]["conditions"]["retrieval_all"]["models"]
    rows = [
        {"title": "Memory retrieval", "scope": "Reused validation · 50 questions",
         "before_label": "Base · full input", "after_label": "Base · target clip",
         "before": count(val["full"]["models"]["base"]),
         "after": count(val["retrieval"]["models"]["base"]),
         "note": "3 more correct answers. Both evidence and prompt change.",
         "source": source_memory["path"],
         "selectors": ["splits.val.conditions.full.models.base",
                       "splits.val.conditions.retrieval.models.base"]},
        {"title": "Memory retrieval", "scope": "Reused test · 27 questions · regression only",
         "before_label": "Base · full input", "after_label": "Base · target clip",
         "before": count(regression["full"]["models"]["base"]),
         "after": count(regression["retrieval"]["models"]["base"]),
         "note": "1 more correct answer. Both evidence and prompt change.",
         "source": source_memory["path"],
         "selectors": ["splits.test.conditions.full.models.base",
                       "splits.test.conditions.retrieval.models.base"]},
        {"title": "Memory-reader tuning", "scope": "Same reused validation · 50 questions",
         "before_label": "Base · retrieved pairs", "after_label": "New memory adapter",
         "before": count(tuned["base"]), "after": count(tuned["tuned"]),
         "note": "No extra accuracy gain. Same evidence and prompt.",
         "source": source_reader["path"],
         "selectors": ["splits.val.conditions.retrieval_all.models.base",
                       "splits.val.conditions.retrieval_all.models.tuned"]},
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "spatial-recall-results-v2"})
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150, facecolor="white")
    fig.subplots_adjust(left=0.285, right=0.965, top=0.80, bottom=0.24)
    fig.suptitle("Memory retrieval and fine-tuning: measured accuracy", x=0.04,
                 y=0.96, ha="left", fontsize=19, weight="bold", color=INK)
    for key, offset, color, legend in (("before", -0.18, MUTED, "Before"),
                                       ("after", 0.18, TEAL, "After")):
        positions = [index + offset for index in range(len(rows))]
        ax.barh(positions, [100 * row[key]["accuracy"] for row in rows],
                height=0.30, color=color, label=legend, zorder=3)
        for y, row in zip(positions, rows):
            value = row[key]
            percent = 100 * value["accuracy"]
            percent_text = f"{percent:g}%" if percent.is_integer() else f"{percent:.1f}%"
            ax.text(2, y, f"{percent_text}  ·  {value['correct']}/{value['total']}",
                    va="center", color="white", fontsize=11, weight="bold", zorder=4)
    ax.set_yticks(range(3), ["Memory retrieval\nValidation · 50 questions",
                           "Memory retrieval\nReused test · 27 questions",
                           "Additional fine-tuning\nValidation · 50 questions"], fontsize=12, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"],
                 fontsize=11, color=MUTED)
    ax.set_xlabel("Answer accuracy", fontsize=12, color=INK, labelpad=9)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e3e6e8", linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2, frameon=False, fontsize=11)
    fig.text(0.04, 0.10, "Retrieval: full history → relevant frames. Fine-tuning: same retrieved evidence, new adapter.",
             fontsize=10.5, color=MUTED)
    fig.text(0.04, 0.055, "AI-reviewed labels · reused cohorts · results do not establish performance on new videos.",
             fontsize=10.5, color=MUTED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        metadata = {"Date": None} if suffix == "svg" else {"Software": "Matplotlib"}
        fig.savefig(OUTPUT / f"results-overview.{suffix}", dpi=120, metadata=metadata)
    plt.close(fig)
    outputs = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                "bytes": path.stat().st_size} for path in sorted(OUTPUT.glob("results-overview.*"))]
    receipt = {"schema": "results-overview-v1", "renderer": "scripts/render_overview.py",
               "renderer_sha256": sha256(Path(__file__)), "matplotlib_version": matplotlib.__version__,
               "sources": [source_memory, source_reader], "comparisons": rows,
               "outputs": outputs, "new_predictions": 0,
               "scope": "Presentation of preserved numerical snapshots; no new training or evaluation."}
    (OUTPUT / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"outputs": outputs, "provenance": "results/overview/provenance.json"}, indent=2))


if __name__ == "__main__":
    main()
