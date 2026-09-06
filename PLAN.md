# Design and reproduction

This document preserves the completed Project 2 fine-tuning contract and describes the subsequent [external-memory refactor](#external-memory-refactor). [REPORT.md](REPORT.md) is the current combined result; [PROGRESS.md](PROGRESS.md) retains the dated audit history. Use a new output directory for a new experiment and preserve the completed run.

To inspect the included evidence without training or installing the ML stack, start with the [current Matplotlib chart](results/overview/results-overview.png) and [actual video-test results](README.md#actual-video-test). The optional [local explorer](#rebuild-the-results-presentation) and original [test](results/fullstudy/figures/test-accuracy.svg) and [training](results/fullstudy/figures/training-dynamics.svg) figures retain the earlier study detail.

## Original experiment contract

| Decision | Implemented choice |
|---|---|
| Task | A–D spatial QA about an earlier clip in supplied, concatenated history |
| Model | `Qwen/Qwen3-VL-4B-Instruct` |
| Model/processor revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Input | Eight reviewed RGB frames with an explicit clip map |
| Intervention | LoRA on language attention `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Frozen weights | All original weights, including vision encoder and mergers |
| Training input | Original training examples only |
| Split caps | 300 train / 50 validation / 50 test; eligibility determines actual counts |
| Retained cohort | 108 train / 50 validation / 27 test |
| Checkpoint choice | End-of-run checkpoint, selected before either model's test scoring |
| Primary effect | Tuned minus base accuracy under competing later footage |
| Supporting comparisons | Original accuracy; neutral minus competing; visual minus text-only |
| Uncertainty | 10,000 paired source/donor-cluster resamples, seed 42 |

Training settings: one epoch, seed 42, batch size one, gradient accumulation four, learning rate `1e-5`, rank 32, alpha 64, dropout 0.05, AdamW, cosine schedule, warmup ratio 0.03, weight decay 0.01 and gradient clipping 1.0. The two-hour cumulative training cap is checked between accumulation groups; a group and final checkpoint can finish after the limit. The completed run reached the end of its epoch.

## System design

The historical fine-tuning application is a single Python batch process per stage, with files connecting stages. The external-memory extension below adds embedded SQLite storage; neither workflow requires a database service. The optional results explorer is a separate static HTML/CSS/JavaScript view over saved outputs; it does not load the model or call an inference API.

| Module | Input → output | Main invariant |
|---|---|---|
| `data.py` | Pinned annotations/media → prepared records and conditions | Earlier evidence is visible; reviews and split rules pass |
| `model.py` | Record + condition → tensors, answer loss or A–D scores | One shared prompt/processor; gold and audit metadata stay out of input |
| `train.py` | Approved original training records → adapter and training state | Original model frozen; only declared adapters train |
| `evaluate.py` | Model state + approved records → atomic prediction rows | Identity matches on resume; failures cannot become valid answers |
| `analysis.py` | Complete paired predictions → metrics and report | All two-model/four-condition cells exist for each test question |
| `artifacts.py` | Config, code, records and files → hashes and frozen protocol | Changes invalidate an existing run's identity |
| `cli.py` | Explicit stage arguments → validated stage execution | Model selection precedes final test scoring |

A prepared record contains an ID, split, question, four options, reviewed answer, source group, original/donor provenance, review decisions and condition definitions. Each visual condition names PNG paths, clip ranges and synthetic timing. Test records include original, neutral, competing and text-only definitions. Review identities bind decisions to content; editing an answer, boundary or donor requires a new review.

Eight frames are assigned as 4+4 for two clips, 4+2+2 for three clips, or 2+2+2+2 for four clips. Sampling stays inside reviewed boundaries. Adjacent prepared frames form temporal pairs; supplied timing describes this prepared sequence, not continuous flight time. Images are resized once to the reviewed canvas, with dimensions divisible by 32. The processor does not resample or resize them again. Overlength inputs are rejected rather than silently truncating visual tokens.

Training masks prompt positions and supervises the answer continuation. Fixed-choice scoring uses a candidate-shaped causal forward whose A–D answer-token scores were checked against the reference implementation. Deterministic SDPA math kernels resolved an observed backward-reproducibility failure; prefix-only BF16 scoring was rejected after a reference mismatch. These decisions and failed diagnostics remain in the full handoff.

For each test question, neutral and competing variants replace one whole later clip. The target frames, question, options, clip map and encoded geometry remain fixed. Connected source/donor components preserve known reuse within bootstrap samples and enforce conservative split separation. Unknown provenance remains a limitation.

## Dataset setup

Use the upstream sources and their terms:

| Source | Pinned revision |
|---|---|
| [SIS-Motion-54K annotations](https://huggingface.co/datasets/choucsan/SIS-Motion-54K) | `6f34a53c77d1709f7494c88833c0ce37e029913e` |
| [SIS-Bench annotations/media](https://huggingface.co/datasets/choucsan/SIS-Bench) | `da50acd6e80cf27413caa27cf90aaf9a6d04e16b` |
| [Training media on ModelScope](https://modelscope.cn/datasets/choucisan/SIS-Motion-54K-Dataset) | `edc34ea44f58a538622ef4221cf5a3b0fe3070a8` |

From an installed source checkout, retrieve and normalize annotation snapshots first:

```bash
export PYTHONPATH=.
python scripts/acquire_data.py --data-root data --annotations-only
python scripts/prepare_full_dataset.py --data-root data --dataset all --plan-only
```

To prepare a bounded batch of up to 50 candidates, omit `--plan-only` and set `--max-items 50`. Preparation is resumable and records failures, held boundaries, source hashes and sampled frames. It does not approve a dataset. The completed study processed all 929 selected training and 152 new test candidates across bounded batches.

Review the source footage, clip boundaries, target visibility at final resolution, viewpoint, and uniqueness of the answer. Record a named reviewer, reviewer type and content identity. Review competing/neutral donors and target preservation separately. The completed cohort required primary and secondary label reviews, held-candidate outcomes, condition reviews and source links. All of those reviews were AI reviews.

| Script | Responsibility |
|---|---|
| `prepare_full_dataset.py` | Selectively decode, sample, fingerprint and queue review |
| `preview_held_training.py` | Display boundary-held candidates for review |
| `recover_reviewed_boundaries.py` | Apply explicit boundary decisions bound to source hashes |
| `review_full_study.py` | Display review identities or apply explicit decisions |
| `build_full_conditions.py` | Build proposed controls from a supplied donor plan |
| `assemble_full_study.py` | Collect reviews, apply vetoes and enforce source separation |

Use each script's `--help` for its paths and arguments. The full-study assembler also expects the original priority queue and review ledgers. With the completed handoff restored, its finalization command is:

```bash
python scripts/assemble_full_study.py --finalize --output data/fullstudy
```

This writes `data/fullstudy/approved.jsonl`, `data/fullstudy/study-audit.json` and `configs/fullstudy.yaml`. It refuses incomplete review gates. A fresh checkout does not contain the handoff, and repeating acquisition does not recreate the original review judgments automatically. To reproduce the exact reported run, restore its frozen inputs and review ledgers. To study new data, create a separately reviewed and frozen run.

## GPU setup

Verified target: Linux x86_64, Python 3.10, one NVIDIA A100 80 GB and a compatible NVIDIA driver. The measured 11.08 GiB peak allocated memory is specific to the final inputs and settings; it is not a guarantee that every 12 GB GPU can run the project.

In a new Python environment on the GPU host, from the project directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[gpu,dev]'
python -m pip install numpy==2.2.6 Pillow==12.0.0 av==13.1.0 PyYAML==6.0.3 huggingface_hub==0.36.0
export HF_HOME="$PWD/.model-cache"
export PYTHONPATH=.
```

The runtime retrieves the base model at the configured immutable revision. The [recorded GPU package lock](results/fullstudy/gpu-lock.txt) and [runtime metadata](results/fullstudy/environment.json) describe the actual completed environment. GPU setup needs network access and storage for the model, data and checkpoints.

`scripts/setup_gpu.sh`, `run_fullstudy.sh`, `run_feasibility.sh` and `archive_gpu.sh` resolve the checkout from their script location; `ECQA_ROOT` can override it. GPU helpers preserve an explicit `HF_HOME`, defaulting to the checkout's ignored `.model-cache/` directory. The setup helper requires `uv` on `PATH`; the pip commands above provide an alternative. Run wrappers retain their named historical output directories, so use the CLI below with fresh output paths for a new experiment. No helper deletes or stops a cloud instance.

The backup helper requires a destination outside the checkout, for example `bash scripts/archive_gpu.sh /external/backup-directory`. It writes `ecqa-backup-inventory.json`, `ecqa-backup.tar.gz` and `ecqa-backup-receipt.json`; use an unused destination to preserve earlier backups. Stored historical source archives retain their original helper scripts.

## Train, evaluate and report

These commands assume `data/fullstudy/approved.jsonl` and every referenced prepared image have been restored and verified. Use the unused `artifacts/reproduction/` directory for a new run:

```bash
python -m ecqa.cli freeze --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/reproduction/protocol.json
python -m ecqa.cli train --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/reproduction/protocol.json --output artifacts/reproduction/train
```

Training writes `summary.json` in its output directory and `selected-model.json` beside the frozen protocol. Read the final checkpoint path from `artifacts/reproduction/train/summary.json`:

```bash
ECQA_ADAPTER="artifacts/reproduction/train/$(python -c "import json; print(json.load(open('artifacts/reproduction/train/summary.json'))['checkpoint'])")"
python -m ecqa.cli evaluate --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/reproduction/protocol.json --output artifacts/reproduction/validation-base --split val
python -m ecqa.cli evaluate --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/reproduction/protocol.json --output artifacts/reproduction/validation-tuned --split val --adapter "$ECQA_ADAPTER"
python -m ecqa.cli evaluate --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/reproduction/protocol.json --output artifacts/reproduction/base
python -m ecqa.cli evaluate --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/reproduction/protocol.json --output artifacts/reproduction/tuned --adapter "$ECQA_ADAPTER"
python -m ecqa.cli report --protocol artifacts/reproduction/protocol.json --manifest data/fullstudy/approved.jsonl --predictions artifacts/reproduction/base/base-test.jsonl artifacts/reproduction/tuned/tuned-test.jsonl --output artifacts/reproduction/report
```

Validation uses original inputs. Test evaluation produces four conditions per question for each model. Neither the final adapter nor the cohort may be selected using test results. The report command needs saved protocol, selection, manifest and predictions, but no GPU or media files.

## Recovery and verification

Full checkpoints contain adapter weights, processor, optimizer/scheduler state, RNG state, data order, configuration identity and file hashes. To resume an interrupted run after a complete update-25 checkpoint:

```bash
python -m ecqa.cli train --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/reproduction/protocol.json --output artifacts/reproduction/train --resume artifacts/reproduction/train/checkpoint-000025
```

Use the actual latest complete checkpoint for that run. An adapter export alone cannot resume training. Only load trusted `training.pt` optimizer-state files. Evaluation resumes existing successful rows by full input/model/protocol identity. Never edit a frozen manifest to get past an integrity error.

To regenerate the completed study's full report after restoring the handoff:

```bash
python -m ecqa.cli report --protocol artifacts/fullstudy/protocol.json --manifest data/fullstudy/approved.jsonl --predictions artifacts/fullstudy/base/base-test.jsonl artifacts/fullstudy/tuned/tuned-test.jsonl --output artifacts/fullstudy-report-regenerated
```

The existing report was regenerated on the Mac and matched the GPU output byte-for-byte. CPU checks cover semantic data validation, prompt isolation, condition preservation, frozen inputs, interrupted predictions and paired analysis. GPU checks cover causal loss, scoring agreement, frozen base weights, adapter updates, save/reload and controlled training resume. The completed run's encoded preflight checked all 185 examples.

## Rebuild the results presentation

Presentation builders run from the project directory and use the included numerical snapshots. They do not train or evaluate a model.

The explorer builder uses only Python's standard library:

```bash
python3 scripts/inspect_results.py
python3 scripts/build_explorer.py
python3 -m http.server 8765 --bind 127.0.0.1
```

Open [http://127.0.0.1:8765/explorer/](http://127.0.0.1:8765/explorer/), then stop the server with **Ctrl+C**. `explorer/index.html` is self-contained: the builder combines `page.html`, `style.css`, `app.js`, metrics, the run/training summaries and training history. It checks snapshot hashes, matching protocol identities and completeness of the recorded training steps before writing the HTML. Edit the three source files and rebuild instead of editing the generated HTML. Direct file opening is supported by design, but browser file mode has not been tested.

The comparison controls display saved full-cohort accuracy or paired change. Question filters affect the tiles and answer details; CSV export contains all 27 questions for the selected condition. The training slider selects recorded optimizer steps and shows the learning rate actually used, rather than the scheduler's next-step rate. The viewer does not accept new questions or produce new predictions. Its schematics illustrate the design; they are not study footage.

The figures can be regenerated separately in the [CPU development environment](CONTRIBUTING.md#set-up-a-cpu-development-environment), with one optional plotting dependency:

```bash
python -m pip install -r requirements-figures.txt
python scripts/render_figures.py
python scripts/render_overview.py
```

The overview builder uses standard Matplotlib horizontal bars and writes SVG/PNG files plus source hashes under `results/overview/`. It shows base retrieval on validation and the reused regression set, followed by the separate new memory-reader candidate. The README states the original distraction-test outcome separately. The original figure builder writes `test-accuracy.svg`, `test-accuracy.png`, `training-dynamics.svg`, `training-dynamics.png` and `provenance.json` under `results/fullstudy/figures/`. The renderer uses the included training history when the original private event log is absent. Provenance records source hashes, renderer versions and output hashes; rendering on another environment is not a promise of byte-identical image files. Matplotlib is optional and is not required for package tests or inspecting the saved figures. The README uses plain Markdown, static result figures and a video attachment; the optional explorer is not required.

The static figures and text remain available directly in the README. GitHub's [markup rendering pipeline](https://github.com/github/markup) removes scripts, so interactive behavior runs in the local HTML page. No public deployment or hosted live demo is required or claimed.

## Reproduce the actual video test

The [recorded walkthrough in the README](README.md#actual-video-test) uses one fixed validation video and two related, previously AI-reviewed questions. Its [six fresh GPU results](results/video-demo/data.json), [frozen protocol](results/video-demo/protocol.json) and [preparation receipt](results/video-demo/preparation.json) are separate from all historical experiment snapshots. The first question preserves the original approved wording; it is not the clarified wording used in the older command-line demonstration. The adapter is the newer memory-reader candidate, not the original full-history adapter. No new training or held-out evaluation is involved.

The README links the owner-authorized 35-second recording as a GitHub attachment; the numerical outputs remain in the repository. Video files, frames, GIF and poster are excluded from Git and source distributions. See [video rights](NOTICE.md#video-footage) and the [distribution receipt](results/video-demo/distribution.json). To reproduce inference, install the CPU package for preparation and the [GPU environment](#gpu-setup) for scoring. Restore the completed memory-reader checkpoint from the verified handoff; model weights and checkpoints are not included in Git. Acquire the pinned MP4 using `source.url` in `data.json`, subject to the source terms, and save it as `artifacts/video-demo/input/source.mp4`. Its required SHA256 is `00fb1953fa10157fa4382b3eb00630eb562f61742863bab91311826de754a708`.

Prepare a new run directory on the canonical decoder platform:

```bash
python scripts/run_video_demo.py --phase prepare \
  --source artifacts/video-demo/input/source.mp4 \
  --output artifacts/video-demo/reproduction
```

The CPU phase commits both questions before decoding, verifies all eight sampled PNG hashes, observes four pairs without questions, seals the memory and constructs reader inputs without gold labels. Source frames are 43, 86, 172, 215, 301, 344, 430 and 473 at 24 fps. Clip 2 retrieval selects frames 172 and 215. The model's prepared-frame timing is synthetic; it is not the raw source frame rate.

This run used macOS arm64, PyAV 13.1.0 and Pillow 12.0.0. Linux decoding reported the same package versions but produced small pixel differences; the runner correctly rejected them before scoring. See the [measured decoder difference](results/video-demo/decoder-comparison.json). Do not bypass the hash check. Reproduce the canonical prepared pixels on the recorded platform, or restore the verified prepared handoff. Exact pixels across operating systems are not guaranteed.

Transfer the prepared directory and the same source checkout losslessly to the GPU host. From that checkout, run:

```bash
python scripts/run_video_demo.py --phase score \
  --adapter artifacts/memory-reader-v1/train/checkpoint-000027 \
  --output artifacts/video-demo/reproduction
```

This freezes the adapter and scoring protocol, loads base and tuned readers in separate processes, scores six cases and writes `data.json`. It checks Question 1 against the independent scorer in full-input and memory-base modes. Full-frame paths are unavailable during memory scoring, and the sealed store must remain byte-identical. Frame selection and prompt are the same for both memory readers; the full-input comparison changes both. Every A–D score, timestamp, model identity and result is saved. Run directories are not overwritten.

Copy the completed directory back locally. Build the browser recording stage from the verified media and results:

```bash
python scripts/build_video_demo.py \
  --source artifacts/video-demo/input/source.mp4 \
  --frames artifacts/video-demo/reproduction/full \
  --results artifacts/video-demo/reproduction/data.json
python3 -m http.server 8766 --bind 127.0.0.1
```

Open [the local recording stage](http://127.0.0.1:8766/output/playwright/video-demo/). It contains no inference code or remote runtime dependencies. Use **Play source footage**, **Show actual GPU answers**, **Show remembered frame**, then **Question 2**. The shorter presentation uses plain answer words; exact tested wording remains available in the expandable details and saved JSON. The remembered frame is the actual source frame at 8.96 seconds. The page labels saved replay and sampled model evidence explicitly. The recording used Playwright browser capture at 1600×900, then FFmpeg H.264 conversion trimmed to 35 seconds by removing the redundant final still; the optional GIF is an accelerated overview; its speed is recorded in the provenance receipt. The [historical provenance receipt](results/video-demo/provenance.json) records output hashes, encoding and browser checks. Stop the local server with Ctrl+C afterward. Source media, the recording stage, MP4, GIF and poster stay in ignored local storage. The README links the authorized recording as a GitHub attachment alongside the measured-answer table. It does not add footage to Git or source distributions. Recreating the recording locally does not grant permission for other uses of its source footage.

## Artifact availability

| Included in the source/result snapshot | Retained in the full local handoff |
|---|---|
| Application, scripts, configs, tests and documentation | Full approved manifest and review/source ledgers |
| Saved report, per-question numerical metrics and checksums | Prepared PNGs, candidate evidence and retrieval references |
| Training history/summary, checkpoint checks and GPU environment metadata | Final adapter, complete resumable checkpoints and diagnostics |
| Offline explorer, SVG/PNG figures, provenance and presentation builders | Frozen protocol, individual predictions and verified backup archive |

The completed adapter is at `artifacts/fullstudy/train/checkpoint-000027/` in the handoff and requires the pinned base model. It has not been uploaded to a model hub. The small `results/fullstudy/` snapshot supports inspecting the numerical result; it cannot independently retrain the model or verify the original visual labels.

The original adapter and the separate memory-reader candidate are both preserved locally. The owner plans to remove the temporary GPU after the verified [local handoff](#restore-after-gpu-removal). Detailed operational receipts stay in ignored maintainer records; no instance deletion was performed.

## External-memory refactor

The workflow separates observation storage, question-time retrieval and model reading. The completed diagnostic reused the pinned base model and existing `artifacts/fullstudy/train/checkpoint-000027/` adapter, with no weight updates during that diagnostic. Its isolated `artifacts/memory-v2/` run completed **2,388 predictions across 362 sealed stores**. A separate [reader-training candidate](#memory-reader-training) also completed; its validation tie retained the base reader.

This is an episodic visual-recall diagnostic on the existing 108 training / 50 validation / 27 previously used test questions. It does not turn those test questions into a fresh holdout or establish continuous 3D spatial memory. The historical four-condition results above remain unchanged.

### Architecture and data contracts

| Module | Responsibility and boundary |
|---|---|
| `memory.py` | Bounded, chronological RGB-pair writes into SQLite plus lossless PNGs; sealing verifies and freezes metadata, retained observations and image hashes |
| `retrieval.py` | Question-only parsing of explicit clip ordinals and selection of retained pairs; a separate oracle function accepts audit targets only for diagnostics |
| `memory_model.py` | Resolve selected IDs from a verified store, preserve original clip identities, and reuse the existing A–D likelihood scorer; plain full-input records retain the original prompt |
| `memory_eval.py` | Atomic scores, strict resume, complete paired reports, target coverage after retrieval, storage/latency and answer-token likelihood |
| `memory_study.py` | CLI orchestration: write and seal all study stores before revealing questions; freeze cases/configuration/code; run separate base/tuned evaluations |
| `memory_train.py` | Freeze one fresh-base reader candidate on train/validation only, train once and select by strict validation improvement |
| `spatial.py` | Validate optional supplied boxes and derive conservative relations within each frame; no automatic detector, tracking or 3D reconstruction |

An observation contains `episode_id`, `scene_id`, one-based `clip_index`, chronological `pair_index`, a timestamp, `source_id` and exactly two RGB frame paths. Scene and source identities remain constant within a supplied clip. Writer inputs exclude questions, options, answers and target annotations. Retention policies are recent/FIFO, seeded reservoir, and causal clip-balanced `episode`; capacity counts retained pairs across the store.

`capacity_pairs` bounds retained RGB pairs, not total bytes or per-episode progress metadata. Prefer one store per episode and measure actual store bytes separately.

A reader record contains only `question`, four `options`, and a `memory` reference containing `episode_id`, `store_digest` and selected `observation_ids`. Gold answers and audit targets stay in scoring metadata. The runtime reopens and verifies the sealed store; unavailable or cross-episode references fail. Empty retrieval supplies no visual tensor. The reader supports at most four pairs; the study normally selects one pair without resizing or synthesizing missing evidence.

The study projects the previously reviewed eight-frame samples into stores. Its timestamps describe their synthetic prepared timeline, not measured flight time. Sampling was fixed by clip boundaries, independent of the answer. Separate question records receive separate episode IDs; the study matrix alone does not demonstrate several new questions against one shared memory. The public `observe`/`answer` interface below supports that experiment without rewriting the store.

### Frozen comparisons

`configs/memory.yaml` fixes seed 42, storage capacity four pairs, reader budget one pair, and nine primary conditions:

| Condition | Supplied evidence / interpretation |
|---|---|
| `full` | All eight frames with the historical prompt; exact baseline |
| `memory_full` | The same four pairs with the memory prompt; isolates prompt/mapping changes |
| `retrieval` | One retained pair from the clip explicitly named in the question |
| `oracle` | One pair selected using the audit target; diagnostic only |
| `oracle_all` | All available target-clip pairs, up to four; diagnoses evidence sufficiency |
| `recent` | Latest retained pair |
| `uniform` | Deterministic uniform selection of one retained pair |
| `wrong` | One pair from another clip within the same video; this is a wrong-clip control |
| `empty` | Question/options with no selected visual observations |

Training receives only the fixed `full` comparison before/after the existing adapter: 108 unchanged examples, accuracy and mean answer-token negative log-likelihood in nats. That likelihood excludes EOS and is not the earlier training-loss trace.

Validation uses all nine conditions plus `episode_capacity2`, `recent_capacity2` and `reservoir_capacity2`, each retaining only two pairs before question-time retrieval. The 27 old test questions receive the nine conditions for original and competing histories. All 1,194 cells per model, 2,388 total, completed and passed snapshot/count verification.

Reports reject missing, duplicate, failed or differently identified cells. They include paired base/tuned changes, prompt/budget contrasts, gold-class counts, clip-delay counts, selected-pair coverage, actual store bytes and separate retrieval/scoring latency. Source-group bootstrap intervals are conditional on available grouping. Target-clip coverage does not prove that a selected pair contains every answer-bearing detail or that the model used it.

Clip-ordinal retrieval is deterministic addressing, not learned semantic search. For questions whose explicit ordinal agrees with the audit, `retrieval` and `oracle` should select identical evidence. A retrieval gain over recent/uniform input would establish a benefit of this selection policy under the tested budget, not a learned global map. Frame count, prompt differences and donor appearance remain explicit diagnostic factors.

### Measured memory diagnostics

| Split / condition | Base | Original adapter |
|---|---:|---:|
| Validation / `full` | 43/50 | 44/50 |
| Validation / `retrieval` | 46/50 | 46/50 |
| Validation / `memory_full` | 41/50 | 42/50 |
| Old test / `full` | 18/27 | 18/27 |
| Old test / `retrieval` | 19/27 | 19/27 |
| Old test / `oracle_all` | 20/27 | 20/27 |
| Old test original / `memory_full` | 16/27 | 17/27 |

The old-test `full`, `retrieval` and `oracle_all` counts also hold for competing histories, each evaluated on the same 27 questions. Retrieval and oracle selected matching target pairs in this diagnostic. Original-adapter training-set accuracy was 96/108 → 95/108; answer-token NLL improved from 0.3560 to 0.3254 nats. These measurements separate likelihood, answer selection and evidence selection; they do not establish better generalization.

The [memory snapshot](results/memory/snapshot.json) includes aggregate metrics, per-question outcomes, runtime receipts, the five-shape GPU scorer/reference check and frozen source. `python3 scripts/inspect_memory_results.py` verifies hashes and recomputes accuracy/cell counts without model or media access. The current implementation passed 251 CPU tests on both local and GPU hosts, plus Ruff. Separate GPU checks exercised two new questions against the same sealed store (B, C for both base/original adapter) and synthetic supplied spatial boxes. They do not establish automatic perception or a 3D map.

### Reproduce or use the memory workflow

Run from an isolated checkout with the verified GPU environment and restored full handoff. Preserve `artifacts/memory-v2/`; the commands below use a separate, unused reproduction directory. To reproduce the completed diagnostic's historical executable, verify the snapshot and extract `results/memory/frozen-source.tar.gz` into an empty checkout. The archive contains source, not model weights, sealed media or the full handoff.

Current code includes later public-CLI safeguards and the training bridge, so its code identity differs from the completed study. A fresh study with current code has a new protocol. Historical resume/reporting requires the matching frozen source; do not disable or rewrite its code checks to make the current checkout impersonate that executable.

```bash
ECQA_MEMORY_RUN=artifacts/memory-reproduction
ECQA_ADAPTER=artifacts/fullstudy/train/checkpoint-000027
python -m ecqa.cli memory prepare --config configs/memory.yaml --manifest data/fullstudy/approved.jsonl --data-root data --output "$ECQA_MEMORY_RUN"
python scripts/memory_gpu_smoke.py --study "$ECQA_MEMORY_RUN"
python -m ecqa.cli memory evaluate --output "$ECQA_MEMORY_RUN"
python -m ecqa.cli memory evaluate --output "$ECQA_MEMORY_RUN" --adapter "$ECQA_ADAPTER"
python -m ecqa.cli memory report --output "$ECQA_MEMORY_RUN"
```

Preparation writes `memory/`, `observation-commit.json`, `cases.json` and `protocol.json`. Evaluation writes separate base/tuned predictions and runtime receipts; reporting writes `report/memory-summary.json` and `report/memory-report.md`. Reporting verifies retained media and therefore needs the sealed stores, unlike the historical numerical-only report.

Use a fresh output directory after interrupted preparation. Once preparation succeeds, evaluation resumes only with the same frozen cases, code, configuration, protocol and model identity. Successful atomic score rows are reused; identical failed rows may be retried. Corrupt artifacts or changed identities require investigation and a distinct run, not edited hashes. `scripts/run_memory_study.sh` wraps these stages for the documented GPU layout; provide an explicit run directory rather than relying on its historical default.

To ask future questions against one memory, first supply observation-only JSONL. Frame paths are relative to that JSONL file. Then supply question JSONL records with exactly `episode_id`, `question` and an A–D `options` object:

```bash
python -m ecqa.cli memory observe --observations observations.jsonl --output artifacts/new-episode --capacity-pairs 4 --policy episode --seed 42
python -m ecqa.cli memory answer --config configs/memory.yaml --store artifacts/new-episode --questions questions.jsonl --output artifacts/new-episode-answers.json --max-pairs 1
```

The answer command requires a supported explicit clip ordinal and refuses missing or ambiguous references. It accepts no benchmark gold. Use another question file and a new answer output path to query the same sealed store again; add `--adapter "$ECQA_ADAPTER"` to use the existing tuned model.

Optional observation `spatial` input contains supplied object annotations with frame-local IDs, frame offset 0/1, labels, normalized boxes and provenance status. `--use-spatial` includes validated annotations during answering; its default is off. Confidence values are uncalibrated, and status values are supplied claims rather than independently verified labels. Only separated image-plane boxes support derived left/right/above/below relations; overlap remains unknown. No automatic detector or cross-view object identity is implemented. The first GPU study includes no supplied spatial annotations and never writes gold answers into memory.

### Memory-reader training

The current bridge trains **one fresh-base LoRA candidate**, rather than continuing the original adapter. It verifies the historical parent cohort and sealed inputs, treats the historical executable as lineage, and freezes the current code/configuration and selected inputs in a new protocol. The candidate output must be new and outside the historical study directory.

Question-only retrieval selects all available target-clip pairs (`retrieval_all`, maximum four); the audit verifies target coverage afterward. Exactly 108 training and 50 validation examples enter this workflow. No old-test examples are materialized or scored. Spatial input is off. Training retains the declared seed-42, learning-rate-`1e-5`, one-epoch settings: 27 optimizer updates with accumulation four.

Run from the current checkout with the completed parent study and original approved manifest restored:

```bash
ECQA_PARENT=artifacts/memory-v2
ECQA_READER=artifacts/memory-reader-reproduction
python -m ecqa.memory_train prepare --study "$ECQA_PARENT" --manifest data/fullstudy/approved.jsonl --output "$ECQA_READER"
python -m ecqa.memory_train base --study "$ECQA_PARENT" --output "$ECQA_READER"
python -m ecqa.memory_train train --study "$ECQA_PARENT" --output "$ECQA_READER"
python -m ecqa.memory_train tuned --study "$ECQA_PARENT" --output "$ECQA_READER"
python -m ecqa.memory_train report --study "$ECQA_PARENT" --output "$ECQA_READER"
```

Run `bash scripts/run_memory_reader.sh "$ECQA_PARENT" data/fullstudy/approved.jsonl "$ECQA_READER"` from the current checkout to wrap the same stages. Memory wrappers preserve `HF_HOME`, default to the checkout's `.model-cache/`, and require the pinned model to be cached because they set offline mode. To resume training, use its `train` phase with `--resume` pointing to a complete checkpoint from this candidate; do not rerun preparation in an existing output directory.

The report verifies the complete final checkpoint and writes `report.json` plus `selected-model.json`. The predeclared rule selects the candidate only when its validation correct count strictly exceeds the matching base on `retrieval_all`; ties or regressions select base. This is one candidate on reused validation, not a hyperparameter search or fresh-test result.

The completed `memory-reader-v1` candidate processed 108/108 examples and 27 updates. Its matched training accuracy was 102/108 → 103/108; validation stayed 46/50 → 46/50. The rule selected **base**. Both sets used all question-selected target pairs, and all 316 before/after predictions completed. [Candidate metrics](results/memory-reader/metrics.json) and [selection receipt](results/memory-reader/selection.json) preserve the outcome. The candidate archive retains the trained adapter for reproducibility; the public CLI's default base reader requires no adapter argument.

### Memory handoff

The local maintainer handoff separates `backups/memory-v2/memory-v2.tar.gz` (sealed evidence, frozen diagnostic and scores) from `backups/memory-reader-v1/memory-reader-v1.tar.gz` (candidate protocol, source, scores, both checkpoints and optimizer states). Each has a SHA256 sidecar and a local verification receipt covering every inventoried file. These ignored archives are not part of the small public numerical snapshot.

Restore archives into an empty destination and preserve their member paths. Pass the restored `memory-v2` directory as `--study` when inspecting or resuming the candidate; it deliberately references the parent stores instead of duplicating them. The candidate's `source.tar.gz` preserves its matching executable for historical use after later code changes. Restore the original approved manifest from the earlier full-study handoff to prepare a new candidate. The pinned base model is a separate dependency available from its recorded Hub revision; these archives do not duplicate the base weights. The owner now plans to remove the temporary GPU; use the verified local handoff described below before doing so. This workflow does not delete the instance.

### Decision gates and research boundaries

1. Preserve the completed scorer/reference, full-input, sealed-store and paired-report evidence. Recheck affected GPU behavior when executable code or input handling changes; CPU tests do not replace those checks.
2. Preserve the completed single-candidate result and its strict validation selection of base. Compare matched reader budgets and separate prompt effects from retrieval, capacity and weight updates. Any further training is a new declared experiment.
3. Before claiming generalization, create a genuinely new source-separated holdout with independent human label validation, predeclared comparisons and adequate scene-level sample size. Existing AI review and repeated-site grouping remain limitations.
4. Before claiming semantic or spatial memory, add unknown future questions against shared stores, a validated semantic retriever, and continuous same-scene video with verifiable cross-view relationships. An RGB-only model may be evaluated against geometric ground truth without receiving those annotations. Do not infer one global coordinate system across unrelated concatenated clips.

These boundaries follow the distinction between [VSI-Bench's spatial reasoning tasks](https://arxiv.org/html/2412.14171v2), [ConceptGraphs' RGB-D/pose-based 3D mapping](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf), and [Q-GeoMem's question-guided geometric memory](https://arxiv.org/html/2605.27318v2). Their architectures and benchmark results are research references, not implemented capabilities or promised improvements in this repository.

## Restore after GPU removal

The 6 September 2026 handoff covers **all 12,238 audited non-cache project files**, totaling 4,266,365,166 bytes across the three GPU project trees. Every remote path maps to a verified local file or archive member, with zero missing files. Both trained adapters, complete optimizer/RNG states, 362 diagnostic memory stores, reviewed inputs, predictions, source code and video-test artifacts are preserved. The larger local dataset also passed its separate **41,152-file** inventory check.

Keep the **whole local project folder**, including ignored `data/`, `artifacts/` and `backups/`. Git alone does not contain the full handoff:

| Local maintainer path | Contents |
|---|---|
| `backups/lax01/` and `backups/lax01-full/` | Verified feasibility and full-study archives |
| `backups/memory-v2/` and `backups/memory-reader-v1/` | Verified memory diagnostic and new-reader training archives |
| `backups/final-handoff/` | Remaining unique files, `coverage.json`, `restore.py` and detailed `RESTORE.md` |
| `artifacts/gpu-retirement/` | Independent verification, remote inventory, runtime and model-availability receipts |

After copying this folder to a future GPU machine, the standard-library restore helper recreates the original three project trees and verifies every written file. Run from the copied repository, using a destination with at least 5 GB free:

```bash
python3 backups/final-handoff/restore.py --destination /ephemeral --dry-run
python3 backups/final-handoff/restore.py --destination /ephemeral
```

All coverage mappings and archive contents were verified. Representative files from all four archives and a direct local file were restored and checked twice. A full extraction and new GPU training-resume test were not performed during this handoff. Use each run's frozen source for historical resume; use the current checkout for a new experiment.

**The handoff requires internet to recreate the runtime.** The 8,887,292,732-byte public base-model cache and disposable virtual environments were excluded; the laptop did not have space for another copy of the base weights. The exact Qwen revision in [GPU setup](#gpu-setup) was available on Hugging Face on 6 September, and published weight hashes plus downloaded configuration/index hashes matched the GPU cache. `backups/final-handoff/RESTORE.md` contains the pinned download command. Allow additional storage for the base model and packages, and populate the cache before using wrappers that enable offline mode. Future upstream availability is not guaranteed by these backups.
