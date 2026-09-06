"""Render source-bound scientific figures without loading a model or using a GPU.

Run from any directory: python scripts/render_figures.py
Optional plotting dependency: pip install -r requirements-figures.txt
The public training-history snapshot suffices when private raw logs are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

ROOT = Path(__file__).resolve().parents[1]
INK = "#172b38"
BASE = "#275dce"
TUNED = "#17847c"
REGRESSION = "#db8032"
MUTED = "#60737d"
GRID = "#dfe7eb"
CONDITIONS = ("original", "neutral", "competing", "text_only")
EVENT_FIELDS = {
    "step": "Completed optimizer iteration, starting at 1; includes the initial zero-LR warmup step.",
    "examples_seen": "Cumulative training examples consumed in the single shuffled epoch.",
    "mean_loss": "Arithmetic mean of per-example supervised-token cross-entropy losses in this group (nats).",
    "gradient_norm": "Total gradient norm returned by clipping, before the clip is applied.",
    "lr_used": "Learning rate used by this optimizer step, recorded before optimizer.step().",
    "lr": "Scheduler learning rate after this optimizer step; this is the next step's rate.",
    "tokens": "Full encoded sequence length for each example in the accumulation group.",
    "elapsed_seconds": "Cumulative training-loop wall time at this event; excludes earlier setup and data work.",
    "peak_vram_bytes": "Peak CUDA memory allocated by PyTorch so far; not total device memory usage.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def relative_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def validate_events(events: list[dict], training: dict) -> None:
    if not events or len(events) != training["updates"]:
        raise ValueError("Event count differs from the saved completed-step count")
    previous_examples, previous_seconds = 0, 0.0
    for step, row in enumerate(events, 1):
        if set(row) != set(EVENT_FIELDS) or row["step"] != step:
            raise ValueError("Unexpected event fields or nonconsecutive training steps")
        for field in set(EVENT_FIELDS) - {"tokens"}:
            value = row[field]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid event field: {field}")
        count = row["examples_seen"] - previous_examples
        if count <= 0 or count != len(row["tokens"]):
            raise ValueError("Token counts disagree with examples consumed in a training group")
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in row["tokens"]):
            raise ValueError("Encoded token counts must be positive integers")
        if row["elapsed_seconds"] < previous_seconds:
            raise ValueError("Training elapsed time goes backward")
        previous_examples, previous_seconds = row["examples_seen"], row["elapsed_seconds"]
    if previous_examples != training["examples_seen"]:
        raise ValueError("Final logged example count differs from the saved training summary")
    if training["stop_reason"] != "epoch_complete" or previous_examples != training["examples_total"]:
        raise ValueError("These completed-epoch figures require a fully completed saved epoch")


def load_history(results: Path, events_path: Path, metrics: dict) -> dict:
    training_path = results / "training.json"
    training = read_json(training_path)
    if training["protocol_digest"] != metrics["identity"]["protocol_digest"]:
        raise ValueError("Training and evaluation belong to different frozen protocols")
    destination = results / "training-history.json"
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        validate_events(events, training)
        history = {
            "schema_version": 1,
            "source": {"path": relative_name(events_path), "sha256": sha256(events_path)},
            "training_summary_source": {"path": relative_name(training_path), "sha256": sha256(training_path)},
            "protocol_digest": training["protocol_digest"],
            "training_identity": training["identity"],
            "field_descriptions": EVENT_FIELDS,
            "events": events,
        }
        if destination.exists() and read_json(destination) != history:
            raise ValueError("Existing public training history differs from the source logs")
        write_json(destination, history)
    else:
        history = read_json(destination)
        if history.get("schema_version") != 1 or history.get("field_descriptions") != EVENT_FIELDS:
            raise ValueError("Unsupported public training-history schema")
        if (history["protocol_digest"] != training["protocol_digest"]
                or history["training_identity"] != training["identity"]
                or history["training_summary_source"]["sha256"] != sha256(training_path)):
            raise ValueError("Public training history is not bound to this training summary")
        validate_events(history["events"], training)
    return history


def configure_matplotlib():
    # Keep optional local font/cache files outside the research artifacts.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ecqa-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none", "svg.hashsalt": "earlier-clip-qa-fullstudy-v1",
        "path.simplify": False,
    })
    return matplotlib, plt


def clean_axes(ax, *, horizontal_grid=False):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y" if horizontal_grid else "x", color=GRID, linewidth=0.7)


def accuracy_figure(plt, metrics: dict):
    from matplotlib.lines import Line2D
    cohort = metrics["cohort"]
    n = cohort["test_questions"]
    clusters = cohort["source_donor_clusters"]
    fig = plt.figure(figsize=(12.8, 7.0))
    fig.text(0.065, 0.93, "TEST RESULTS", fontsize=10, weight="bold", color=MUTED)
    fig.text(0.065, 0.871, "Accuracy before and after fine-tuning", fontsize=23, weight="bold")
    fig.text(0.065, 0.821,
             f"Qwen3-VL-4B-Instruct  ·  {n} test questions  ·  {clusters} supplied clusters  ·  one training seed",
             fontsize=11.4, color=MUTED)
    ax = fig.add_axes((0.20, 0.265, 0.615, 0.445))
    clean_axes(ax)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Test accuracy (%)", labelpad=12)
    ax.set_ylim(-0.55, 3.55)
    ax.set_yticks([3, 2, 1, 0], ["Original", "Neutral", "Competing", "Text only"])
    ax.tick_params(axis="y", pad=18, labelsize=12)
    ax.axhspan(0.54, 1.46, color="#f3f7f8", zorder=0)
    ax.text(-0.055, 0.63, "primary", transform=ax.get_yaxis_transform(), ha="right",
            fontsize=9, color=MUTED)
    ax.text(1.035, 1.10, "Accuracy · correct/total", transform=ax.transAxes,
            fontsize=9.2, color=MUTED)
    for index, condition in enumerate(CONDITIONS):
        center = 3 - index
        for model, shift, color, marker in (("base", 0.13, BASE, "o"), ("tuned", -0.13, TUNED, "s")):
            cell = metrics["accuracies"][model][condition]
            value, bounds = 100 * cell["accuracy"], cell["ci95"]
            if cell["total"] != n or cell["accuracy"] != cell["correct"] / n:
                raise ValueError("Saved accuracy and counts disagree")
            if bounds is not None:
                if not 0 <= bounds[0] <= cell["accuracy"] <= bounds[1] <= 1:
                    raise ValueError("Invalid saved confidence interval")
                ax.errorbar(value, center + shift,
                            xerr=[[value - 100 * bounds[0]], [100 * bounds[1] - value]],
                            fmt="none", ecolor=color, elinewidth=1.65, capsize=4, capthick=1.3, alpha=0.7)
            ax.plot(value, center + shift, marker=marker, markersize=8.3, color=color,
                    markeredgecolor="white", markeredgewidth=1, zorder=4)
            ax.text(1.035, center + shift, f"{value:.1f}%  ·  {cell['correct']}/{n}",
                    transform=ax.get_yaxis_transform(), va="center", color=color, fontsize=10.8)
    handles = [Line2D([], [], color=BASE, marker="o", linestyle="none", markersize=7, label="Base"),
               Line2D([], [], color=TUNED, marker="s", linestyle="none", markersize=7, label="LoRA tuned")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.20, 0.789), ncol=2,
               frameon=False, handletextpad=0.45, columnspacing=1.8)
    primary = metrics["effects"]["competing_gain"]["estimate"]
    neutral = metrics["accuracies"]["tuned"]["neutral"]["correct"] - metrics["accuracies"]["base"]["neutral"]["correct"]
    formatted_gain = f"{100 * primary:+.2f}" if primary else "0.00"
    fig.text(0.065, 0.144, f"Competing accuracy change: {formatted_gain} percentage points.",
             fontsize=11, weight="bold")
    fig.text(0.56, 0.144, f"Neutral: {neutral:+d} correct answer.", fontsize=11,
             color=REGRESSION if neutral < 0 else INK)
    fig.text(0.065, 0.100,
             "Error bars: 95% accuracy intervals from paired source/donor-cluster bootstrap "
             f"({metrics['bootstrap']['completed_samples']:,} resamples).",
             fontsize=9.5, color=MUTED)
    fig.text(0.065, 0.069,
             "Intervals are conditional on supplied grouping; source independence is unverified. These are not intervals for the paired gain.",
             fontsize=9.3, color=MUTED)
    fig.text(0.065, 0.038,
             "Small, AI-reviewed custom cohort; one seed. No demonstrated population improvement or memory mechanism.",
             fontsize=9.3, color=MUTED)
    return fig


def training_figure(plt, history: dict):
    from matplotlib.ticker import ScalarFormatter
    events = history["events"]
    steps = [row["step"] for row in events]
    losses = [row["mean_loss"] for row in events]
    rates = [row["lr_used"] for row in events]
    fig = plt.figure(figsize=(12.8, 7.4))
    fig.text(0.075, 0.946, "TRAINING EVIDENCE", fontsize=10, weight="bold", color=MUTED)
    fig.text(0.075, 0.891, "One completed epoch, shown without smoothing", fontsize=22, weight="bold")
    fig.text(0.075, 0.846,
             f"{events[-1]['examples_seen']} examples  ·  {len(events)} optimizer iterations  ·  "
             f"{sum(rate > 0 for rate in rates)} iterations with a nonzero learning rate  ·  LoRA on language attention",
             fontsize=11, color=MUTED)
    loss_ax = fig.add_axes((0.095, 0.505, 0.845, 0.255))
    rate_ax = fig.add_axes((0.095, 0.184, 0.845, 0.225), sharex=loss_ax)
    for ax in (loss_ax, rate_ax):
        clean_axes(ax, horizontal_grid=True)
        ax.set_xlim(0.65, len(events) + 0.35)
        ax.set_xticks([1, 5, 10, 15, 20, 25, len(events)])
    loss_ax.plot(steps, losses, color=BASE, linewidth=1.4, marker="o", markersize=4.6,
                 markeredgecolor="white", markeredgewidth=0.6)
    loss_ax.set_ylim(0, max(losses) * 1.12)
    loss_ax.set_ylabel("Mean loss (nats)", labelpad=13)
    loss_ax.set_title("A   Recorded mean loss per accumulation group", loc="left", fontsize=11,
                      weight="bold", pad=15)
    loss_ax.tick_params(labelbottom=False)
    rate_ax.plot(steps, rates, color=TUNED, linewidth=1.7, marker="o", markersize=4,
                 markeredgecolor="white", markeredgewidth=0.5)
    rate_ax.set_ylim(-0.04 * max(rates), 1.10 * max(rates))
    rate_ax.set_ylabel("Learning rate used", labelpad=13)
    rate_ax.set_xlabel("Optimizer iteration", labelpad=11)
    rate_ax.set_title("B   Rate actually used by each optimizer step", loc="left", fontsize=11,
                      weight="bold", pad=20)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-5, -5))
    rate_ax.yaxis.set_major_formatter(formatter)
    rate_ax.set_yticks([0, 0.25e-5, 0.5e-5, 0.75e-5, 1e-5])
    if rates[0] == 0:
        rate_ax.plot(steps[0], 0, marker="o", color=REGRESSION, markersize=5.5, zorder=5)
        rate_ax.annotate("Step 1: LR used = 0\n(warmup)", xy=(1, 0),
                         xytext=(3.6, 0.23 * max(rates)), fontsize=9.3, color=REGRESSION,
                         arrowprops={"arrowstyle": "-", "color": REGRESSION, "lw": 0.9})
    fig.text(0.075, 0.068,
             "Raw recorded values; no smoothing. Each group contains different examples, so this trace does not establish convergence.",
             fontsize=9.4, color=MUTED)
    fig.text(0.075, 0.037,
             "Loss averages supervised-token cross-entropy per example. LR uses the logged lr_used field, not the scheduler's next-step rate.",
             fontsize=9.3, color=MUTED)
    return fig


def save_figure(fig, output: Path, name: str, description: str) -> dict:
    paths = {}
    for extension in ("svg", "png"):
        path = output / f"{name}.{extension}"
        metadata = ({"Title": name.replace("-", " ").title(), "Description": description,
                     "Creator": "earlier-clip-qa / Matplotlib", "Date": None} if extension == "svg"
                    else {"Software": "earlier-clip-qa / Matplotlib", "Description": description})
        fig.savefig(path, dpi=180, metadata=metadata)
        paths[extension] = {"path": relative_name(path), "sha256": sha256(path)}
    return {"description": description, "files": paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/fullstudy")
    parser.add_argument("--events", type=Path, default=ROOT / "artifacts/fullstudy/train/events.jsonl")
    parser.add_argument("--output", type=Path, help="Defaults to <results>/figures")
    args = parser.parse_args()
    output = args.output or args.results / "figures"
    metrics_path = args.results / "metrics.json"
    metrics = read_json(metrics_path)
    history = load_history(args.results, args.events, metrics)
    matplotlib, plt = configure_matplotlib()
    output.mkdir(parents=True, exist_ok=True)
    cohort = metrics["cohort"]
    count_summary = "; ".join(
        condition.replace("_", " ") + ": " + " and ".join(
            f"{metrics['accuracies'][model][condition]['correct']}/{cohort['test_questions']}"
            for model in ("base", "tuned"))
        for condition in CONDITIONS)
    events = history["events"]
    descriptions = {
        "test-accuracy": f"Base and tuned accuracy on {cohort['test_questions']} questions: {count_summary}. "
                         "The accuracy axis covers 0–100%. Error bars are conditional 95% paired "
                         "source/donor-cluster bootstrap accuracy intervals from "
                         f"{metrics['bootstrap']['completed_samples']:,} resamples of "
                         f"{cohort['source_donor_clusters']} supplied clusters.",
        "training-dynamics": f"All {len(events)} recorded optimizer-group losses and actual learning rates "
                             f"for one epoch over {events[-1]['examples_seen']} examples. "
                             f"{sum(row['lr_used'] > 0 for row in events)} steps use a nonzero learning rate. "
                             "Values are unsmoothed. Changing examples across groups mean the training "
                             "trace does not establish convergence.",
    }
    figures = {}
    for name, figure in (("test-accuracy", accuracy_figure(plt, metrics)),
                         ("training-dynamics", training_figure(plt, history))):
        figures[name] = save_figure(figure, output, name, descriptions[name])
        plt.close(figure)
    import numpy
    provenance = {
        "schema_version": 1,
        "sources": {"metrics": {"path": relative_name(metrics_path), "sha256": sha256(metrics_path)},
                    "training_history": {"path": relative_name(args.results / "training-history.json"),
                                         "sha256": sha256(args.results / "training-history.json")},
                    "raw_events": history["source"]},
        "protocol_digest": metrics["identity"]["protocol_digest"],
        "renderer": {"path": relative_name(Path(__file__)), "sha256": sha256(Path(__file__)),
                     "python": platform.python_version(), "matplotlib": matplotlib.__version__,
                     "numpy": numpy.__version__, "png_dpi": 180},
        "figures": figures,
    }
    write_json(output / "provenance.json", provenance)
    print(json.dumps({"status": "rendered", "training_events": len(history["events"]),
                      "figures": list(figures), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
