"""Build a portable, offline HTML explorer from verified saved study results."""
import hashlib
import json
from pathlib import Path


def build(root):
    source = root / "results" / "fullstudy"
    checks = json.loads((source / "checksums.json").read_text())
    for name, expected in checks["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("Snapshot path leaves the result directory")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected["sha256"] or len(content) != expected["bytes"]:
            raise ValueError(f"Result snapshot checksum mismatch: {name}")
    payload = {key: json.loads((source / name).read_text()) for key, name in (
        ("metrics", "metrics.json"), ("summary", "run-summary.json"),
        ("training", "training.json"), ("history", "training-history.json"))}
    protocol = payload["summary"]["protocol_digest"]
    if any(value != protocol for value in (
        payload["metrics"]["identity"]["protocol_digest"],
        payload["training"]["protocol_digest"], payload["history"]["protocol_digest"])):
        raise ValueError("Explorer inputs belong to different experiments")
    events = payload["history"]["events"]
    if [row["step"] for row in events] != list(range(1, payload["training"]["updates"] + 1)):
        raise ValueError("Training history is incomplete")
    if events[-1]["examples_seen"] != payload["training"]["examples_seen"]:
        raise ValueError("Training history and summary disagree")
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    serialized = serialized.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    folder = root / "explorer"
    html = (folder / "page.html").read_text()
    for marker, value in (("__EXPLORER_CSS__", (folder / "style.css").read_text()),
                          ("__EXPLORER_DATA__", serialized),
                          ("__EXPLORER_JS__", (folder / "app.js").read_text())):
        if html.count(marker) != 1:
            raise ValueError(f"Template requires exactly one {marker}")
        html = html.replace(marker, value)
    output = folder / "index.html"
    output.write_text(html)
    print(json.dumps({"output": str(output.relative_to(root)), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "test_questions": payload["metrics"]["cohort"]["test_questions"],
                      "training_events": len(events), "external_runtime_dependencies": 0}))


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
