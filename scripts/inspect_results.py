"""Check and display the saved study results using only the standard library.

This reads recorded outputs. It does not run inference or repeat the experiment.
"""

import hashlib
import json
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parents[1] / "results" / "fullstudy"
    checks = json.loads((folder / "checksums.json").read_text())
    for name, expected in checks["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Result snapshot path leaves its directory")
        content = path.read_bytes()
        if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"Saved result checksum mismatch: {name}")
    metrics = json.loads((folder / "metrics.json").read_text())
    training = json.loads((folder / "training.json").read_text())
    summary = json.loads((folder / "run-summary.json").read_text())
    questions = metrics["per_question"]
    count = len(questions)
    if len({row["id"] for row in questions}) != count or count != metrics["cohort"]["test_questions"]:
        raise ValueError("Saved question count is inconsistent")
    if count * 8 != metrics["cohort"]["prediction_cells"]:
        raise ValueError("Saved prediction count is inconsistent")
    if summary["protocol_digest"] != training["protocol_digest"]:
        raise ValueError("Saved training and summary identities differ")

    print("Earlier-clip QA — saved results (no model inference)")
    print(f"Training: {training['examples_seen']}/{training['examples_total']} examples; "
          f"{training['updates']} optimizer steps; {training['stop_reason']}")
    print(f"Test: {count} questions; {count * 8} model/condition predictions")
    print(f"{'Condition':<15} {'Base':>12} {'Tuned':>12} {'Change (pp)':>13}")
    for condition in ("original", "neutral", "competing", "text_only"):
        counts = []
        for model in ("base", "tuned"):
            correct = sum(row["conditions"][condition][f"{model}_prediction"] == row["answer"]
                          for row in questions)
            saved = metrics["accuracies"][model][condition]
            if saved["correct"] != correct or saved["total"] != count or saved["accuracy"] != correct / count:
                raise ValueError(f"Saved accuracy differs from per-question outcomes: {model}/{condition}")
            counts.append(correct)
        base, tuned = (f"{value}/{count}" for value in counts)
        change = 100 * (counts[1] - counts[0]) / count
        print(f"{condition:<15} {base:>12} {tuned:>12} {change:>+13.2f}")
    validation = summary["validation"]
    print("Validation: " + " → ".join(f"{validation[model]['correct']}/{validation[model]['rows']}"
                                        for model in ("base", "tuned")))
    print("Fine-tuning executed successfully. Primary test accuracy did not improve.")
    print("Snapshot hashes and reported accuracy counts agree.")
    print("Limits: AI labels, 27 test questions, one training seed, uncertain source independence.")


if __name__ == "__main__":
    main()
