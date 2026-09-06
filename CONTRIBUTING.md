# Contributing

Start with the [README](README.md) for project scope and [REPORT.md](REPORT.md) for the current combined results. Contributions that make the experiment easier to understand, reproduce, or verify are useful, including documentation corrections and small regression tests.

## Set up a CPU development environment

Run these commands from the repository root, the folder containing `pyproject.toml`. Use Python 3.10 on macOS or Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ecqa --help
```

The base package and `dev` extra install the CPU dependencies and development tools. PyTorch, Transformers, model weights, dataset downloads, and a GPU are unnecessary for the test suite. Installation itself needs access to the Python package index. Synthetic tests create their own PNGs and small videos in temporary directories, including real codec checks through PyAV.

Run the checks from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
python -m ruff check ecqa tests scripts
python -m pip check
```

The [CPU workflow](.github/workflows/tests.yml) uses these checks on Ubuntu with Python 3.10. It also checks the installed CLI, saved result snapshots, Git media exclusions and tests from the built source archive. GitHub Actions discovers this workflow at the standalone repository root. Check the [hosted run history](https://github.com/laxman-kc/spatialcontext/actions/workflows/tests.yml) for each published commit's outcome.

For workflow edits, also run `actionlint .github/workflows/tests.yml`. YAML parsing alone does not validate GitHub Actions expression contexts.

For diagram edits, render the Mermaid blocks in both `README.md` and `ARCHITECTURE.md` with the [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli) before publishing. Link checks and CPU tests do not validate diagram syntax. Avoid literal semicolons in sequence-diagram message text: Mermaid treats them as statement separators.

## Make a focused change

- Keep the flat `ecqa/` package and thin CLI. Follow the responsibilities documented in the README.
- For a behavioral fix, add a small test that reproduces the failure. Prefer generated fixtures to real dataset files. Keep ordinary tests independent of network access, model caches, and CUDA.
- Document any change to prompts, frame selection, scoring, split rules, review gates, or checkpoint selection. These changes require a new experiment identity when used for research.
- Keep REPORT.md, the README overview and explorer entry panel consistent with the verified metrics; identify each adapter and evaluation split. Keep recorded results and frozen inputs intact. A new experiment belongs in a separate run directory with its own protocol; changes to historical claims should explain the evidence and preserve the original record.
- Keep credentials, downloaded media, model weights, checkpoints, local paths, and large generated artifacts out of contributions. `data/`, `artifacts/`, and `backups/` are ignored intentionally. A small result snapshot should retain its provenance and limitations.

GPU diagnostics and research runs are separate from CPU CI. Use the [GPU workflow](PLAN.md#gpu-setup) only when the runtime, data review, and compute budget are ready. Changes to encoding or training need the relevant processor/backward/save-reload checks before claiming GPU correctness; a passing CPU suite alone cannot establish that.

## Describe the contribution

For a bug report, include the command, expected behavior, actual result, Python/OS versions, and a small reproducible example. Remove tokens and private paths from logs. For a pull request, explain the problem, resulting behavior, and checks actually run. Documentation-only changes can use a link and command review instead of unrelated tests.

Keep scientific claims tied to the saved evidence. Distinguish software correctness from model quality, AI review from human validation, and an observed cohort result from a population claim. A negative result is a valid outcome. Consult the README for the current licensing and release status before redistributing code or upstream assets.

## Prepare a push

Original project code is MIT-licensed; preserve [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). The README links the owner-authorized recording as a GitHub attachment. Its MP4, raw footage, frames, GIF and poster remain excluded from Git and source distributions. Keep those exclusions; permission for this demonstration does not cover other footage.

Work from the standalone repository's `main` branch. It begins with a clean root commit so earlier media-containing commits are not part of the public history. The former `codex/release-ready` branch and private Git bundle retain local history; do not push them. Include source, tests, documentation and the small `results/` snapshots. Keep original media, memory databases, model weights, private credentials and backup archives ignored. The two small frozen-source archives under `results/` are intentional reproducibility inputs. `.gitattributes` preserves the exact bytes of checksum-bound results.

Run the CPU checks above and verify the saved evidence:

```bash
python3 scripts/inspect_results.py
python3 scripts/inspect_memory_results.py
python3 scripts/inspect_memory_results.py --root results/memory-reader
git diff --check
git diff --cached --stat
git status --short
git remote -v
```

`origin` points to [laxman-kc/spatialcontext](https://github.com/laxman-kc/spatialcontext). Publish only the reviewed `main` branch:

```bash
git push -u origin main
```

Do not use `--all` or `--mirror`: local private branches contain excluded footage. The remote is configured for this project; pushing `main` publishes only its reviewed history. The source ZIP is another history-free distribution option. Model and dataset terms remain separate from the MIT license.
