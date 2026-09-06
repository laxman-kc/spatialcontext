# Spatial recall from video — current results, 6 September 2026

The original question was whether task-specific fine-tuning improves answers about earlier spatial observations when later footage contains competing landmarks. The completed primary distraction test remained **18/27 before and after fine-tuning**. The subsequent memory experiment tested a separate intervention: retaining and retrieving the relevant earlier frames.

**Selected system: Qwen3-VL-4B base reader with persistent memory retrieval.** Retrieving the relevant earlier clip improved measured accuracy. The separately trained memory-reader adapter did not add validation accuracy, so the declared selection rule retained base.

## Before and after memory retrieval

These rows compare the **same base model** with full prepared history versus the requested clip's retained frame pair.

| Evaluation | Without retrieval | With retrieval | Net change |
|---|---:|---:|---:|
| Validation, 50 questions | 43/50 · 86% | **46/50 · 92%** | **+3 answers / +6 percentage points** |
| Previously used test, 27 questions | 18/27 · 66.7% | **19/27 · 70.4%** | **+1 answer / +3.7 percentage points** |

Retrieval changes both the supplied images and prompt formatting. A same-frame prompt control scored 41/50 on validation, so simply using the memory prompt did not reproduce the retrieval gain. The old test is reused regression evidence, not a fresh holdout. Its competing-footage variant also scored 18/27 → 19/27. [Exact diagnostic metrics](results/memory/metrics.json).

An **annotation-assisted all-target oracle** reached 20/27 on the old test. That is a separate diagnostic upper reference; it is not the normal one-pair retrieval result or a new test of the newly trained adapter.

## What fine-tuning achieved

| Separate training experiment | Before | After | Interpretation |
|---|---:|---:|---|
| New memory-reader candidate: validation | 46/50 | 46/50 | No extra accuracy gain; select base |
| New memory-reader candidate: training fit | 102/108 | 103/108 | One additional training answer correct |
| Original full-history adapter: primary test | 18/27 | 18/27 | Historical result; no primary gain |

The new candidate completed 108 examples and 27 optimizer updates. All 288 saved LoRA tensors changed between checkpoints 25 and 27, and all 158 train/validation score vectors changed. Training therefore ran and updated the adapter; unchanged validation choices explain the unchanged accuracy. [Training audit](results/memory-reader/numerical-audit.json) · [Selection receipt](results/memory-reader/selection.json).

The `tuned` column in the earlier memory diagnostic refers to the **original full-history adapter**. The candidate metrics refer to a **different, newly trained memory-reader adapter**. They must not be treated as one model or one experiment.

## Actual video demonstration

Six real GPU predictions were recorded with the source footage and retained frames. The public release preserves the numerical evidence; the 35-second recording is retained only in the local maintainer handoff because footage redistribution permission was not established.

| Fixed question | Base, full history | Base, retrieved frames | New tuned reader, same retrieved frames |
|---|---|---|---|
| Object right of the bag-loaded truck | White car — incorrect | Spherical lights — correct | Spherical lights — correct |
| Truck's position relative to the lights | Left — correct | Left — correct | Left — correct |

This is one previously used validation video and two related AI-reviewed questions. It demonstrates the interface and one correction, not a new benchmark. [Exact tested wording, scores and timestamps](results/video-demo/data.json).

## Scope and source reports

Memory persists on disk, is written before questions, and retrieves explicit clip numbers. These experiments do not establish semantic retrieval, human-like recall, automatic object tracking, 3D mapping or generalization to unseen videos. Labels were AI-reviewed. No result in this report is an assumed improvement.

The following reports preserve their own completed experiments and remain byte-identical to their verified snapshots:

- [Original fine-tuning report](results/fullstudy/report.md): historical 27-question primary test.
- [Memory diagnostic report](results/memory/report.md): 2,388 predictions across 362 stores; its tuned model is the original adapter.
- [New memory-reader metrics](results/memory-reader/metrics.json): 316 predictions, with validation-only selection of the new candidate.

Use this document for the combined current result. [Concise README](README.md) · [Progress and audit history](PROGRESS.md) · [Reproduction and local restore](PLAN.md#restore-after-gpu-removal).
