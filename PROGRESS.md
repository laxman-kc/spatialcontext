# Progress and measured evidence

**Current result: retrieval improved base validation from 43/50 to 46/50 and reused-test accuracy from 18/27 to 19/27. The new memory-reader adapter stayed at 46/50, selecting base. [Current combined report](REPORT.md). All audited GPU project files are preserved locally; the public base model must be downloaded again for future GPU use. [Restore instructions](PLAN.md#restore-after-gpu-removal).**

## Public-release preparation — 6 September 2026 UTC

Publication uses **`main`** at [laxman-kc/spatialcontext](https://github.com/laxman-kc/spatialcontext). The owner supplied this destination and requested `main`; the private historical branch is excluded from the push.

The first hosted workflow stopped before creating any jobs. `actionlint` identified an unsupported `runner.temp` reference in job-level environment configuration. Cache initialization now uses `$RUNNER_TEMP` in a workflow step, where the runner environment is available; the corrected workflow passes `actionlint`. Hosted outcomes are recorded in the linked GitHub Actions run history.

The owner selected MIT for original project code. `LICENSE`, package license metadata and `NOTICE.md` now state that scope and preserve dataset/reference attributions. The public release excludes the third-party MP4, GIF and poster, including their earlier Git versions. The recordings and old Git history remain in the local handoff; the public `main` branch has no private-history ancestor.

The README retains the original spatial-recall question, all measured outcomes and a plain Matplotlib chart. The actual video test is shown as a table linked to its unchanged questions, scores, timestamps and provenance. No model weights or dataset media are bundled. The numerical snapshots and frozen source archives retain their verified bytes.

A fresh CPU installation exposed a source-distribution packaging defect: the archive omitted scripts and configurations needed by two test modules. `MANIFEST.in` now includes the source workflow, tests, configs, documentation and numerical results, while excluding footage. The rebuilt source archive passed all 251 CPU tests, Ruff and all three saved-result inspectors. A separately installed wheel passed CLI startup, dependency checks and an import check outside the source folder. Both artifacts include MIT/Apache notices; archive inspection found no footage, model weights or ignored private data. The CI workflow now also checks media exclusions and tests the source distribution. Local command logs and package inventories are under ignored `artifacts/public-release/`. These local checks preceded GitHub publication and involved no new GPU experiment. Hosted results are available in the [CPU workflow](https://github.com/laxman-kc/spatialcontext/actions/workflows/tests.yml).

## Current documentation and final GPU handoff — 6 September 2026 UTC

README clarification: the descriptive title is now **Spatial recall from video**. It leads with the original fine-tuning/distraction question and its unchanged 18/27 result, then explains the measured memory improvement. The current figure uses standard Matplotlib bars with exact counts on a shared 0–100% scale. The README is plain Markdown with static images; the optional HTML explorer is no longer part of its quickstart. At that review, public release still required a code license and a decision on the third-party footage; the public-release preparation above resolves these by adding MIT and excluding footage.

The README, results image and explorer now lead with the same current memory comparisons. [REPORT.md](REPORT.md) combines the measured retrieval gain, separate fine-tuning outcome and actual video test. The five earlier Project 2 planning/research files outside this repository now carry a current-status link and historical labels. Frozen experimental reports retain their original bytes and are identified as source evidence, avoiding any change to verified measurements.

All **12,238 audited non-cache GPU files / 4,266,365,166 bytes** have verified local coverage, with zero missing files. Only 38 missing unique files, 1,988,178 bytes, needed downloading; other additional files were already local and copied into the final handoff. All four existing archive inventories passed SHA-256/size verification, including both adapters and resumable training states. A separate check verified all **41,152 local dataset files / 11,820,350,011 bytes**. The handoff includes 362 diagnostic memory stores and the actual video-test inputs and outputs.

The restore helper resolved every covered path. Representative files from each archive and a direct local object passed restoration, hash checks and a second identical restore. Full extraction and a new GPU resume were not performed. Receipts and inventory are under ignored `artifacts/gpu-retirement/` and `backups/final-handoff/`; preserve the whole local project, including its ignored data and backups.

The 8.89 GB public base-model cache and disposable runtime caches were excluded. Its exact pinned Hub revision was reachable, and weight hashes plus downloaded configuration/index hashes matched the GPU cache. Future execution requires downloading that base model and reinstalling packages. The owner plans GPU removal; this handoff performed no stop or delete operation. [Restore and dependency details](PLAN.md#restore-after-gpu-removal).

## Git preparation — 6 September 2026

The initial standalone repository used `codex/release-ready`, with its README, package and CI workflow at the root. That preparation contained 101 files, approximately 2.1 MB. A later private presentation added an annotated recording, preview and evidence receipts; the public branch subsequently excluded the footage. Original downloaded media, runtime memories, checkpoints, credentials, caches and full backup archives remain excluded. The small, integrity-bound source archives under `results/` are included intentionally.

A fresh export of the staged Git files passed **251 CPU tests**, Ruff, CLI startup, shell syntax checks and all three result inspectors. The checks used the existing CPU environment while importing the exported source. All local documentation paths resolved. Separate content review found no credentials or unsafe archive members. The local receipt and check logs are under ignored `artifacts/release/`.

Documentation now distinguishes the original study from subsequent memory experiments and dates superseded diagnostic gaps. Current GPU helpers use portable checkout/cache paths, and the backup helper requires an external destination. Frozen results and historical source archives retain their exact bytes. CI now verifies all saved result snapshots in addition to package, lint and CPU checks; a hosted CI run has not been claimed.

The initial Git preparation left the remote URL and original code license as owner choices. The later public release adds the owner's chosen MIT license and a clean public branch, now named `main` for publication. See [push preparation](CONTRIBUTING.md#prepare-a-push).

## Actual video test and recording — 6 September 2026 UTC

Presentation follow-up: the README now leads with one current-results image covering validation retrieval, reused-test retrieval and the new memory-reader candidate. The original primary test remains in the dated historical report. Detailed experiment history stays linked here and in PLAN.md. The demo uses larger answer words, a 35-second recording, and a static clickable preview; exact tested questions and all measured outputs remain unchanged. [Current image and source hashes](results/overview/provenance.json).

The private presentation recorded the actual 21.5-second AirScape_Train_8918 source video. Its [numerical results remain public](README.md#actual-video-test). Two fixed questions were committed before scoring: the original approved validation wording and a previously reviewed inverse relation question. Six fresh predictions ran on the A100 using the pinned base model and the new memory-reader candidate; no additional training was performed.

The first answer changed **A (incorrect) → B (correct)** when switching from eight-frame full input to two-frame memory retrieval; the tuned memory reader also returned B. The second answer was C, correct, in all three conditions. Thus retrieval corrected one demonstrated answer, while the adapter supplied no additional accuracy gain. Evidence and prompt both change in the first comparison; only the adapter changes in the second. These two related, AI-reviewed questions from reused validation footage do not establish generalization.

All eight newly decoded canonical PNG hashes matched the reviewed evidence. Initial Linux decoding of the same MP4 produced differences of at most three RGB channel levels and was rejected before inference. Local Mac decoding reproduced the exact historical pixels; those frames and the sealed memory were transferred losslessly to the GPU. Both installations reported PyAV 13.1.0; the precise cause beyond the platform difference was not isolated. The [decoder comparison](results/video-demo/decoder-comparison.json) records this limitation instead of silently accepting different model inputs.

Question 1 reference checks matched exactly in full-input and memory-base modes (maximum score difference 0.0); these checks do not separately cover Question 2 or the adapter. Memory inventory hashes remained identical across fresh base/adapter processes, and original full-frame paths were unavailable during memory scoring. The [saved data](results/video-demo/data.json) includes each A–D score vector, timestamps, measured durations and model identities. Saved JSON timings include input preparation and scoring, excluding model loading; they are not a throughput benchmark.

The recording is an explicitly labeled browser replay of the real run. It shows source playback, the retained target pair, both questions and their actual outcomes. [Provenance and presentation checks](results/video-demo/provenance.json) bind the recording to the measured data. The GPU was checked after completion: no compute processes and zero allocated GPU memory; the instance remains intact. The original footage retains third-party rights, with its individual source-owner mapping unresolved; no external upload is claimed.

## Memory refactor — 6 September 2026 UTC

The reader can now reopen a sealed SQLite/PNG memory and answer multiple later questions using retained evidence. Observation writing is chronological and question-blind; retrieval uses the requested clip ordinal. Frame retention supports episode coverage, recent observations and seeded reservoir sampling. Missing retained evidence causes the public CLI to refuse an answer. Questions do not modify the sealed store.

Optional supplied bounding boxes support image-plane relations such as left/right and above/below. This is not object detection, cross-frame tracking, a persistent 3D map, or a simulation of human memory. The current dataset contains concatenated clips without a verified common coordinate system.

The completed diagnostic sealed **362 stores** and produced **2,388 predictions**: 1,194 each for the base reader and the original adapter. It includes a same-frame prompt control, question-selected evidence, oracle evidence, recent/uniform/wrong/empty evidence, competing later clips and reduced memory capacities. The numerical snapshot is in [results/memory](results/memory); `python3 scripts/inspect_memory_results.py` verifies its hashes and recomputes every accuracy count.

| Evidence condition | Validation: base / original adapter | Old regression set: base / original adapter |
|---|---:|---:|
| All eight frames, original prompt | 43/50 / 44/50 | 18/27 / 18/27 |
| All eight frames, memory prompt | 41/50 / 42/50 | 16/27 / 17/27 |
| Question-selected target pair | 46/50 / 46/50 | 19/27 / 19/27 |
| All target pairs, oracle diagnostic | 46/50 / 46/50 | 20/27 / 20/27 |
| Most recent pair | 17/50 / 17/50 | 10/27 / 10/27 |
| Uniformly selected pair | 35/50 / 35/50 | 16/27 / 16/27 |
| No images, memory prompt | 26/50 / 26/50 | 11/27 / 10/27 |

The regression column uses the original clips. Target retrieval and all-target oracle scores are identical under the competing-later-clip variant. All conditions, source-group intervals, corrected/regressed cases, evidence coverage, storage sizes and measured timings are in [the diagnostic report](results/memory/report.md).

**What this establishes:** relevant evidence selection improves this development cohort; applying the original adapter to the retrieved evidence gives no accuracy gain. Merely changing the prompt does not explain the retrieval improvement: the same-frame memory prompt performs worse. Clip-ordinal retrieval matches the oracle because the questions explicitly identify the target clip. Even all target evidence leaves four validation and seven regression errors, so retention alone cannot explain every failure. These checks do not distinguish remaining perception, reasoning and label errors.

**What this does not establish:** generalization to a fresh test set, reliable open-ended retrieval, long-stream spatial mapping, or causal evidence use for each correct answer. Labels were reviewed by AI, and the old test set is now an exposed regression set. Above-chance empty-evidence accuracy is another reason not to equate correct choices with demonstrated visual recall. At a two-pair storage capacity, validation accuracy is 31/50 for episode retention, 30/50 for recent retention and 38/50 for reservoir retention; policy and capacity are part of the result.

### Verification and reproducibility

- **251 CPU tests and Ruff passed** locally and on the GPU host. The frozen diagnostic source had 219 tests; later public CLI guards and the training bridge account for the current release count.
- GPU scoring matched the unchanged reference exactly for 0, 2, 4, 6 and 8 frames. The original six model/data/train/evaluate/analysis/artifact modules remain byte-identical to the diagnostic baseline.
- Two AI-reviewed questions about one retained real clip were answered correctly by both the base and original adapter in separate public CLI processes. All ten sealed-store files remained byte-identical afterward. These correlated questions demonstrate integration, not a new benchmark; receipts are in `artifacts/memory-release-evidence/public-demo/`.
- A synthetic supplied-box spatial test passed on the A100 with an exact reference-score match after removing its original source image. It tests annotation-to-reader integration, not real-world spatial detection.
- The diagnostic protocol is `163c9999da2f2e65c486b2e9547787d34e948cd7b26697766b785e48f039b716`. [The frozen source archive](results/memory/frozen-source.tar.gz) preserves its executable code. Current code has a separate identity; historical reproduction must use the archived source.
- Base/tuned diagnostic passes took 684.09 / 735.81 seconds after model initialization, including case validation and scoring. Peak allocated GPU memory was 9,753,198,080 / 9,847,569,920 bytes. Setup, memory preparation and loading are additional costs.
- The diagnostic backup was copied to the Mac and verified by streaming every archived file: **5,429 data files / 766,553,129 bytes**, plus its inventory. The compressed archive is 747,577,204 bytes, SHA256 `9d59362a5a738e971bca89175c0ce2b6cbc54c9ca4fdd98bceb7e532bcb6efcc`. Receipt: `backups/memory-v2/verification.json`. No media extraction was needed for verification.

### One memory-reader training candidate

The new candidate starts from the base model and uses question-selected **all-target-pair evidence** for 108 training and 50 validation questions. It retains the declared one-epoch, learning-rate 1e-5, seed-42 schedule. Its final checkpoint is selected only if validation accuracy strictly exceeds the matched base reader; a tie retains the base. There is one candidate and no new scoring of the old test set. This is development on reused AI-reviewed validation, not an independent confirmation.

Candidate protocol: `09ed5345b40203867baed4272e076c246bd2c12ba168c15c0b8a70e63ae6ea29`. All **316 before/after predictions** completed and the wrapper exited with code zero.

| Candidate measure | Base | New memory-trained adapter |
|---|---:|---:|
| Training accuracy | 102/108 | 103/108 |
| Validation accuracy | 46/50 | 46/50 |
| Training answer-token NLL | 0.13396 | 0.12003 |
| Validation answer-token NLL | 0.17962 | 0.18838 |

**Selected: base with memory retrieval.** Training fit improved slightly, while validation accuracy did not improve and validation answer-token likelihood worsened. These measurements do not prove a general cause such as overfitting. The training loop completed 108/108 examples and 27 optimizer updates in 103.09 seconds, with 10,973,631,488 bytes peak allocated GPU memory and 23,592,960 trainable parameters. This timing includes training/checkpointing and excludes preparation, model loading and evaluation.

The independent [numerical/checkpoint audit](results/memory-reader/numerical-audit.json) passed: every score vector changed across the 158 examples, but **none of the 50 validation answer choices changed**. All 288 saved LoRA tensors are finite and changed between checkpoints 25 and 27. All 144 LoRA-B tensors contain nonzero updates, totaling 10,027,008 nonzero values; 26 optimizer steps used a nonzero learning rate. The unchanged validation accuracy is therefore a measured result of an adapter that did update, not evidence that training was skipped.

The [candidate metrics](results/memory-reader/metrics.json), [selection receipt](results/memory-reader/selection.json) and [training record](results/memory-reader/training.json) are separate from the original adapter. `python3 scripts/inspect_memory_results.py --root results/memory-reader` verified all nine snapshot files and recomputed all 316 prediction cells. The archived source matches the frozen candidate code identity.

The candidate backup passed local streaming verification of **360 data files / 601,003,173 bytes**, plus its inventory. It includes both checkpoints with optimizer state, the exact source, environment, protocol, predictions and audit. The compressed archive is 531,494,085 bytes, SHA256 `5c69a633bd235d5b9da124b80bfb8bc95b5a2995fbcd5a147bb4651d78c3e08c`; receipt: `backups/memory-reader-v1/verification.json`. Parent memory stores are preserved in the separate verified diagnostic archive.

At 03:57 UTC on 6 September, the retained `lax01` A100 reported **0 MiB GPU memory, 0% utilization and no compute processes**. The completed candidate exit code was zero. The read-only status receipt is in `artifacts/memory-release-evidence/final-gpu-status.json`. Archives were verified without extracting large files; the Mac had approximately 0.8 GiB free at handoff, so restore them on a volume with sufficient space.

## Original bounded fine-tuning study

**Historical status: implementation, bounded dataset audit, training, evaluation and verified local handoff completed. Fine-tuning executed successfully; primary test accuracy did not improve.**

The implementation is documented in [README.md](README.md), and the experiment contract and reproduction workflow are in [PLAN.md](PLAN.md). The current GPU instance remains intact under the owner's retention instruction.

[Explore saved results locally](PLAN.md#rebuild-the-results-presentation) · [Test accuracy figure](results/fullstudy/figures/test-accuracy.svg) · [Training dynamics figure](results/fullstudy/figures/training-dynamics.svg)

## What has been achieved

| Area | Completed evidence |
|---|---|
| Application | Installable Python package with preparation, freeze, train, evaluate and report stages |
| Dataset | 108 training / 50 validation / 27 test questions after explicit review and source exclusions |
| Training | One full epoch; 108/108 examples; 27 optimizer steps; final `checkpoint-000027` |
| Adapter | 23,592,960 trainable parameters; saved weights changed and passed integrity checks |
| Evaluation | All 216 test cells plus 100 original-only validation predictions |
| Correctness | 115 CPU tests and Ruff passed; GPU loss/scoring/update/reload checks passed |
| Recovery | Controlled uninterrupted-versus-resumed training produced matching states and scores |
| Handoff | GPU archive copied and verified locally; full report reproduced byte-for-byte |
| Onboarding | Visual README, offline results explorer, SVG/PNG figures, CPU quickstart and contributor/CI files |

“Working” means the fine-tuning pipeline trains, saves, reloads and evaluates the adapter correctly on the verified setup. It does not mean this adapter is more accurate on the primary test.

## Measured outcome

| Test condition | Base | Tuned | Difference |
|---|---:|---:|---:|
| Original | 18/27 · 66.67% | 18/27 · 66.67% | 0.00 pp |
| Neutral later clip | 19/27 · 70.37% | 18/27 · 66.67% | −3.70 pp |
| Competing later clip | 18/27 · 66.67% | 18/27 · 66.67% | 0.00 pp |
| Text only | 5/27 · 18.52% | 5/27 · 18.52% | 0.00 pp |

Validation original-only accuracy was **43/50 → 44/50**. The primary competing-condition change was **zero**. The neutral-minus-competing gap shrank because neutral performance worsened by one question; it does not show better resistance to distraction. The predefined promising-result criterion was not met. No further training or parameter selection used these test results.

Only neutral-condition question `test:positional_relationship_0232` changed correctness, from correct to incorrect. Original, competing and text-only correctness was unchanged for every test question. Full numerical outcomes, corrected/regressed IDs and cluster assignments are in [metrics.json](results/fullstudy/metrics.json); uncertainty estimates are in the [paired report](results/fullstudy/report.md).

## Evidence that training ran

The run ended with `stop_reason: epoch_complete`. Mean training loss across the epoch was **0.1183879**; all logged losses and gradients were finite. The adapter contains 288 finite tensors, including 144 LoRA-B tensors with 10,027,008 nonzero elements. Checks verified frozen original weights, adapter updates and save/reload behavior.

The training loop and checkpoint writes took **112.96 seconds**, with **11.08 GiB peak allocated GPU memory** on an A100 80 GB. These measurements exclude setup, data curation, model loading and evaluation. They are not an end-to-end runtime or a minimum hardware specification.

See [training.json](results/fullstudy/training.json), [checkpoint-verification.json](results/fullstudy/checkpoint-verification.json) and [environment.json](results/fullstudy/environment.json). These are copied records from the completed run, not a claim that model weights are included in this source snapshot.

The [training dynamics figure](results/fullstudy/figures/training-dynamics.svg) plots all 27 unsmoothed accumulation-group losses and the learning rate used by each optimizer step. The initial warmup step used zero learning rate; 26 steps used a nonzero rate. Each group contains different examples, so this trace does not establish convergence. The [training history](results/fullstudy/training-history.json) preserves the source event-log hash and descriptions of every plotted field.

## Dataset audit and exclusions

The bounded pool was exhausted: **929 selected training candidates and 152 new test candidates** were processed. The proposed 300/50/50 split sizes were caps, not promised counts.

- Training preparation produced 689 boundary-supported records, 139 held boundaries, 77 decoded-final-target exclusions and 24 unresolved content references.
- Of the held training candidates, 117 required visual review: 16 were recovered and 101 excluded. Another 22 had metadata exclusions: 15 final-clip targets and seven unresolved targets. Secondary review could veto recovered labels.
- Test boundary review recovered 15 held candidates and excluded one ordinal mismatch.
- A fixed-seed second review of 12 initially accepted training/validation labels found seven nonunique or unresolved answers. This triggered a complete second review of potentially selected labels and all 40 proposed controlled test originals.
- All 185 selected labels have content-bound primary and secondary reviews. Thirteen test labels were vetoed on second review, leaving 27 complete paired test questions. All proposed controls were reviewed.
- Known original/donor source overlap is separated across splits. Prior feasibility exposures were excluded from the final test through conservative source grouping.

Reviewers were explicitly recorded as **AI, not human**. Original decisions, vetoes, boundary recoveries, source evidence and exclusions remain in the full local handoff. These are reviewed judgments rather than guaranteed ground truth; conservative review can also exclude valid examples.

## Interpretation limits

The 27 test questions form ten conservative connected source/donor components of sizes 9, 8, 3 and seven singletons. The paired cluster bootstrap uses 10,000 resamples with seed 42. Groups are few and unequal, and unresolved provenance means independence is not established.

Original, competing and text-only paired correctness changes are all zero, giving empirical difference intervals of [0, 0]. Those intervals describe this observed sample; they do not establish population equivalence or a zero population effect.

One training seed, AI labels, imperfect donor matching, uncertain source provenance and unknown upstream model-training contamination limit generalization. The visual conditions outperform text-only on this cohort, but the experiment does not identify a memory mechanism or report a full official SIS-Bench score.

## Reproducibility and handoff

| Identity | Value |
|---|---|
| Protocol digest | `abfff2fb1645b298c389f7ace1d09f09cb425348b601ca26c1a9cc7c1443eb78` |
| Approved manifest SHA256 | `5837ad9a3f29967309f62efe4cd28736e508715c3524b4264678e921b204190c` |
| Selected checkpoint | `checkpoint-000027` |
| Archive SHA256 | `5f6de03a7b18d32d4a890a1651e9fc1134c1a1545e6c2d4e0dab7a5131f7574f` |

The GPU archive contains **5,934 files / 1,528,802,519 uncompressed bytes**; every file passed individual SHA256 verification on the Mac. The compressed archive is 1,404,741,341 bytes. Frozen input and final checkpoint checks passed locally, and the regenerated report and metrics are byte-identical to the GPU originals. A separate audit recomputed all 216 test and 100 validation rows and checked model, protocol and input identities.

The complete local evidence inventory verified 41,625 files / 12,719,430,476 bytes at the training handoff. It is a dated inventory of that handoff; later documentation and packaging changes do not rewrite that historical receipt. Full receipts remain under ignored `backups/lax01-full/`. The working immutable artifacts use hard links to the extracted archive to conserve disk; the compressed archive is an independent copy.

`results/fullstudy/` contains byte-identical copies of the report, metrics, training summary, checkpoint checks and environment records, plus a derived `run-summary.json` and source-bound `training-history.json`. `scripts/inspect_results.py` checks snapshot hashes and recomputes accuracy counts from saved per-question outcomes. The SVG/PNG figures have a separate [provenance record](results/fullstudy/figures/provenance.json) identifying their inputs, renderer and outputs. These files support result inspection, not full data/checkpoint distribution.

## Visual presentation

The [explorer](explorer/index.html) embeds the original study's saved data and runs without external runtime dependencies. It exposes condition comparisons, per-question outcomes, CSV export and all recorded training steps. The README includes static figures and exact-count text; [local launch and regeneration instructions](PLAN.md#rebuild-the-results-presentation) explain how to use the interactive page because GitHub README pages do not execute its JavaScript. The later memory studies have separate numerical snapshots and tables.

The explorer contains saved answer letters and correctness outcomes. It does not include the original question text or footage, run inference, or establish a memory mechanism. Its illustrative clip diagrams are labeled as schematics. This record does not establish completed desktop, mobile, keyboard or export browser checks. Direct browser file mode has not been tested.

## Development history and next research steps

The first engineering feasibility run used 4 training / 1 validation / 1 test example. Its 80-test record, one-update checkpoint, diagnostics and backup remain separate. Later review found nonunique labels in two of its training records, so its results support only historical engineering work. The bounded original study and subsequent memory experiments are reported separately above.

The original documentation and presentation refresh preserved its detailed operational history in ignored maintainer records and did not retrain that adapter. The later memory-reader training run is reported at the top of this document; its separate artifacts preserve the original study. The GPU was retained at that check. No public upload or hosted CI execution was claimed. The owner subsequently selected MIT, as recorded in the public-release section above.

A future improvement study would need a new predeclared experiment, stronger independent label/source review, and validation-driven choices before a fresh test. More training may or may not help. These are research follow-ups, not unfinished steps in the completed bounded experiment.

## Historical diagnosis — 6 September 2026, 01:22 UTC

This audit predates the completed memory refactor and reader candidate. Its original-run observations remain valid; the later experiments above address the then-missing fixed-training-set comparison and subsequent training.

A read-only audit of saved logs and individual predictions confirmed why the accuracy totals stayed unchanged. No new training was performed. The live GPU check at 01:22 UTC found no project training/evaluation jobs, zero GPU utilization and 0 MiB reported GPU memory use; the completed pipeline exit code was zero.

| Evaluation | A–D score vectors changed | Selected answers changed | Correct answers: base → tuned |
|---|---:|---:|---:|
| Validation original | 50/50 | 1 | 43 → 44 |
| Test original | 25/27 | 0 | 18 → 18 |
| Test neutral | 27/27 | 1 | 19 → 18 |
| Test competing | 27/27 | 0 | 18 → 18 |
| Test text-only | 22/27 | 1 | 5 → 5 |

All 27 primary-condition score vectors changed, but every highest-scoring answer stayed the same. That directly explains the unchanged primary accuracy. Across the test, 101/108 score vectors changed and only two selected answers changed. The neutral change was a regression; the text-only change remained incorrect. Validation's single correction resolved a prior A/B score tie in favor of the reviewed answer B.

For neutral question `test:positional_relationship_0232`, the reviewed answer was B, the gray monument. The base preferred B over A by 0.25 log-score units; the tuned model preferred A, the red monument, over B by 0.25. This is an observed ranking reversal. The saved predictions do not establish a perceptual or internal-memory cause for it.

Training completed because the configured single epoch ended after 108 examples. It used 27 optimizer steps, of which 26 had nonzero learning rates; step 1 used zero during the configured one-step warmup. Twenty logged pre-clipping gradient norms exceeded the configured threshold of 1.0. These are measured schedule/clipping facts, not demonstrated causes of the absent accuracy gain.

At the time of this audit, the fixed-training-set comparison before and after training was missing. The later memory diagnostic supplied it: accuracy 96/108 → 95/108 and answer-token NLL 0.3560 → 0.3254. The original recorded batch losses compare different four-example groups and cannot establish convergence, underfitting or overfitting. There is still no controlled sweep of training duration, learning rate, adapter scope, precision or seeds. Dataset review exclusions and the small AI-reviewed cohort are documented limitations; their causal contribution to these model errors is unmeasured.

The supported original-run conclusion remains limited: training changed weights and scores, but the final adapter did not correct any test answer. The exact cause of the absent test improvement remains unresolved. Subsequent work completed the memory retrieval comparison and a validation-only reader candidate, as reported above. Detailed source-bound files from this earlier audit remain under ignored `artifacts/post-training-audit/`.
