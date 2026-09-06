# Earlier-clip QA: paired pilot results

Tuned competing-condition accuracy is equal on this frozen cohort. This is an exploratory observed comparison; it does not establish a memory mechanism, population improvement, or noninferiority.

The complete matrix contains 27 questions and 216 predictions across 10 supplied source groups and 10 connected source/donor clusters.

## Accuracies

| Condition | Base correct/total | Base accuracy (%) [95% CI] | Tuned correct/total | Tuned accuracy (%) [95% CI] |
|---|---:|---:|---:|---:|
| original | 18/27 | 66.67 [47.05, 85.00] | 18/27 | 66.67 [47.05, 85.00] |
| neutral | 19/27 | 70.37 [57.89, 90.00] | 18/27 | 66.67 [50.00, 83.33] |
| competing | 18/27 | 66.67 [45.83, 90.91] | 18/27 | 66.67 [45.83, 90.91] |
| text_only | 5/27 | 18.52 [0.00, 42.86] | 5/27 | 18.52 [0.00, 42.86] |

## Paired effects

All effects and interval bounds below are percentage points. The primary effect is competing_gain.

| Effect | Estimate (pp) | 95% CI (pp) |
|---|---:|---:|
| original_gain | 0.00 | [0.00, 0.00] |
| neutral_gain | -3.70 | [-20.00, 0.00] |
| competing_gain | 0.00 | [0.00, 0.00] |
| text_only_gain | 0.00 | [0.00, 0.00] |
| base_semantic_distraction_penalty | 3.70 | [-7.91, 18.18] |
| tuned_semantic_distraction_penalty | 0.00 | [-20.00, 16.13] |
| semantic_penalty_reduction | 3.70 | [0.00, 20.00] |
| base_original_to_competing_drop | 0.00 | [-20.00, 7.69] |
| tuned_original_to_competing_drop | 0.00 | [-20.00, 7.69] |
| base_visual_benefit | 48.15 | [16.67, 73.08] |
| tuned_visual_benefit | 48.15 | [16.67, 73.08] |

## Corrections and regressions

| Condition | Corrected | Regressed | Unchanged correct | Unchanged incorrect |
|---|---:|---:|---:|---:|
| original | 0 | 0 | 18 | 9 |
| neutral | 0 | 1 | 18 | 8 |
| competing | 0 | 0 | 18 | 9 |
| text_only | 0 | 0 | 5 | 22 |

## Uncertainty and interpretation

Paired source/donor-cluster percentile bootstrap: 10000 resamples, seed 42, question-weighted estimates. Every resample retains all conditions and model predictions together.

These intervals are conditional on the available provenance groups. No independence, causal mechanism, maintained-performance, or publication-level claim follows from the intervals alone.

- Some source provenance is unverified; supplied group labels do not establish independence.
- Intervals are conditional on the available source/donor grouping; independence is not established by the group labels.
- One training run leaves variation across training seeds unmeasured.
- This custom subset and edited-condition diagnostic is not a full official SIS-Bench score.
- Unknown upstream model-training contamination and preparation uncertainty limit generalization claims.
- Recorded test reviewers: 0 human, 27 AI. AI review is not human validation.

The conditional primary-effect interval includes zero; the observed difference remains uncertain under this resampling scheme.

The predefined descriptive promising pattern is not met: higher observed competing accuracy with equal or higher observed original and neutral accuracy. This is a descriptive pilot criterion, not proof of improvement or noninferiority.

## Reproducibility

- Protocol digest: `abfff2fb1645b298c389f7ace1d09f09cb425348b601ca26c1a9cc7c1443eb78`
- Base model digest: `dc6e4377addd21e22e89ffdec6d23c985fdb106601645601d9d88124b2b0208a`
- Tuned model digest: `cc96b49c44a7ccbef17dfbbb85074e36c455b7bc834d575ba7158581d1cceb83`
- Analysis input digest: `bbe92bfcc6f9fbb170391805033e27fb27704a61475635e3f12d398f89fe2ca7`

Question-level paired outcomes, corrected/regressed IDs, answer-label counts, cluster assignments, and all numerical estimates are retained in metrics.json.
