# License scope and acknowledgments

Original project code, documentation and generated result figures are licensed under the [MIT License](LICENSE), copyright 2026 laxkc and contributors. This grant also covers the original project code preserved inside the two `results/*/frozen-source.tar.gz` archives. It does not replace third-party terms.

Package metadata lists `MIT AND Apache-2.0` as the aggregate release terms because the source distribution also includes dataset-derived text. This is not a choice of licenses for every file: original code remains MIT, and the dataset-derived material below retains Apache-2.0. The wheel contains the runtime package and notices; the fuller source archive contains the experiment workflow and numerical evidence.

## Research, models and libraries

- [Self-in-Space](https://github.com/IntelliSensing/Self-in-Space/tree/128a33a0c608493d6f80ec206503ab3ff88136fb), by the IntelliSensing/Self-in-Space authors, informed the task, dataset selection and LoRA settings. Its upstream implementation is Apache-2.0. This repository implements its own training and evaluation workflow; it does not bundle the upstream trainer or a pretrained Self-in-Space model.
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), by the Qwen team, is the separately downloaded base model. Its weights and processor are not distributed here. Follow the model's own license and notices.
- The frame-canvas calculation in `ecqa/data.py` follows [Qwen3-VL processor conventions](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py), with fixed spatial factors and per-frame pixel bounds. This acknowledges algorithm compatibility; it is not a claim that the Qwen implementation is bundled or covered by this project's MIT license.
- PyTorch, Transformers, PEFT and other dependencies are installed separately and retain their respective licenses. No vendored dependency source or model weights are included.

## Dataset-derived text

Question wording, answer choices and source identifiers in the numerical snapshots and video-test runner derive from [SIS-Motion-54K](https://huggingface.co/datasets/choucsan/SIS-Motion-54K) and [SIS-Bench](https://huggingface.co/datasets/choucsan/SIS-Bench), released by the Self-in-Space dataset authors. Their dataset cards identify Apache-2.0 for those resources. A copy is included at [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

This project selected and reviewed a subset, applied documented label/boundary decisions, constructed later-footage controls, and produced predictions and metrics. These are project-specific experiments, not official benchmark scores. The recorded source revisions and review limitations are in [PLAN.md](PLAN.md) and [REPORT.md](REPORT.md). Dataset-derived text retains its upstream terms; the MIT grant does not relicense it.

## Video footage

AirScape's [dataset card](https://huggingface.co/datasets/EmbodiedCity/AirScape-Dataset) reserves original video rights to the source owners. On 6 September 2026, the project owner confirmed permission to publish the existing 35-second annotated recording of AirScape_Train_8918. The [README](README.md#actual-video-test) links that recording as a GitHub video attachment. This confirmation covers this demonstration; it does not grant a blanket license to AirScape footage or place the underlying video under MIT.

The recording remains excluded from Git history and source distributions. Raw footage, sampled frames, the GIF, the poster and earlier private Git history remain in the local maintainer handoff. The [distribution receipt](results/video-demo/distribution.json) identifies the published attachment; the historical provenance and numerical results retain their original hashes. Follow the source owners' terms and obtain any needed permission for other uses.
