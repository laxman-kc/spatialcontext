External-memory diagnostic results

These are training/validation diagnostics and regression checks on a previously used test set.

| Split | Condition | Model | Correct / count | Accuracy | Gold NLL (nats) | Scoring seconds |
|---|---|---|---:|---:|---:|---:|
| test | full | base | 18 / 27 | 0.6667 | 0.8678 | 0.531 |
| test | full | tuned | 18 / 27 | 0.6667 | 0.8658 | 0.535 |
| test | memory_full | base | 16 / 27 | 0.5926 | 0.9759 | 0.760 |
| test | memory_full | tuned | 17 / 27 | 0.6296 | 0.9751 | 0.774 |
| test | retrieval | base | 19 / 27 | 0.7037 | 0.6666 | 0.375 |
| test | retrieval | tuned | 19 / 27 | 0.7037 | 0.6727 | 0.423 |
| test | oracle | base | 19 / 27 | 0.7037 | 0.6666 | 0.392 |
| test | oracle | tuned | 19 / 27 | 0.7037 | 0.6727 | 0.434 |
| test | oracle_all | base | 20 / 27 | 0.7407 | 0.6309 | 0.406 |
| test | oracle_all | tuned | 20 / 27 | 0.7407 | 0.6277 | 0.458 |
| test | recent | base | 10 / 27 | 0.3704 | 1.8476 | 0.381 |
| test | recent | tuned | 10 / 27 | 0.3704 | 1.8069 | 0.428 |
| test | uniform | base | 16 / 27 | 0.5926 | 1.1768 | 0.382 |
| test | uniform | tuned | 16 / 27 | 0.5926 | 1.1771 | 0.430 |
| test | wrong | base | 12 / 27 | 0.4444 | 1.6580 | 0.378 |
| test | wrong | tuned | 12 / 27 | 0.4444 | 1.6420 | 0.432 |
| test | empty | base | 11 / 27 | 0.4074 | 3.5914 | 0.275 |
| test | empty | tuned | 10 / 27 | 0.3704 | 3.6666 | 0.331 |
| test | competing_full | base | 18 / 27 | 0.6667 | 0.7588 | 0.524 |
| test | competing_full | tuned | 18 / 27 | 0.6667 | 0.7597 | 0.540 |
| test | competing_memory_full | base | 17 / 27 | 0.6296 | 0.8565 | 0.768 |
| test | competing_memory_full | tuned | 17 / 27 | 0.6296 | 0.8446 | 0.799 |
| test | competing_retrieval | base | 19 / 27 | 0.7037 | 0.6666 | 0.382 |
| test | competing_retrieval | tuned | 19 / 27 | 0.7037 | 0.6727 | 0.431 |
| test | competing_oracle | base | 19 / 27 | 0.7037 | 0.6666 | 0.385 |
| test | competing_oracle | tuned | 19 / 27 | 0.7037 | 0.6727 | 0.438 |
| test | competing_oracle_all | base | 20 / 27 | 0.7407 | 0.6309 | 0.445 |
| test | competing_oracle_all | tuned | 20 / 27 | 0.7407 | 0.6277 | 0.462 |
| test | competing_recent | base | 9 / 27 | 0.3333 | 1.7921 | 0.382 |
| test | competing_recent | tuned | 9 / 27 | 0.3333 | 1.7910 | 0.458 |
| test | competing_uniform | base | 16 / 27 | 0.5926 | 1.1768 | 0.425 |
| test | competing_uniform | tuned | 16 / 27 | 0.5926 | 1.1771 | 0.441 |
| test | competing_wrong | base | 10 / 27 | 0.3704 | 2.0251 | 0.403 |
| test | competing_wrong | tuned | 10 / 27 | 0.3704 | 2.0056 | 0.436 |
| test | competing_empty | base | 11 / 27 | 0.4074 | 3.5914 | 0.277 |
| test | competing_empty | tuned | 10 / 27 | 0.3704 | 3.6666 | 0.336 |
| train | full | base | 96 / 108 | 0.8889 | 0.3560 | 0.517 |
| train | full | tuned | 95 / 108 | 0.8796 | 0.3254 | 0.555 |
| val | full | base | 43 / 50 | 0.8600 | 0.3469 | 0.516 |
| val | full | tuned | 44 / 50 | 0.8800 | 0.3362 | 0.553 |
| val | memory_full | base | 41 / 50 | 0.8200 | 0.4879 | 0.745 |
| val | memory_full | tuned | 42 / 50 | 0.8400 | 0.4770 | 0.781 |
| val | retrieval | base | 46 / 50 | 0.9200 | 0.1879 | 0.370 |
| val | retrieval | tuned | 46 / 50 | 0.9200 | 0.1884 | 0.431 |
| val | oracle | base | 46 / 50 | 0.9200 | 0.1879 | 0.376 |
| val | oracle | tuned | 46 / 50 | 0.9200 | 0.1884 | 0.421 |
| val | oracle_all | base | 46 / 50 | 0.9200 | 0.1796 | 0.377 |
| val | oracle_all | tuned | 46 / 50 | 0.9200 | 0.1810 | 0.444 |
| val | recent | base | 17 / 50 | 0.3400 | 2.3850 | 0.371 |
| val | recent | tuned | 17 / 50 | 0.3400 | 2.3717 | 0.434 |
| val | uniform | base | 35 / 50 | 0.7000 | 1.3502 | 0.383 |
| val | uniform | tuned | 35 / 50 | 0.7000 | 1.3295 | 0.438 |
| val | wrong | base | 27 / 50 | 0.5400 | 1.9933 | 0.369 |
| val | wrong | tuned | 27 / 50 | 0.5400 | 1.9513 | 0.424 |
| val | empty | base | 26 / 50 | 0.5200 | 2.3337 | 0.268 |
| val | empty | tuned | 26 / 50 | 0.5200 | 2.3566 | 0.325 |
| val | episode_capacity2 | base | 31 / 50 | 0.6200 | 1.7689 | 0.225 |
| val | episode_capacity2 | tuned | 31 / 50 | 0.6200 | 1.7782 | 0.272 |
| val | recent_capacity2 | base | 30 / 50 | 0.6000 | 2.0458 | 0.207 |
| val | recent_capacity2 | tuned | 30 / 50 | 0.6000 | 2.0661 | 0.261 |
| val | reservoir_capacity2 | base | 38 / 50 | 0.7600 | 1.0786 | 0.244 |
| val | reservoir_capacity2 | tuned | 38 / 50 | 0.7600 | 1.0879 | 0.293 |

Gold NLL is answer-token negative log-likelihood, excluding EOS; it is not training loss.

| Split | Condition | Target clip hit rate | Target pair recall | Selected pairs | Store bytes | Retrieval seconds |
|---|---|---:|---:|---:|---:|---:|
| test | full | 1.0000 | 1.0000 | 4.0000 | 2616453.0000 | 0.0000 |
| test | memory_full | 1.0000 | 1.0000 | 4.0000 | 2616453.0000 | 0.1197 |
| test | retrieval | 1.0000 | 0.7963 | 1.0000 | 2616453.0000 | 0.1188 |
| test | oracle | 1.0000 | 0.7963 | 1.0000 | 2616453.0000 | 0.1191 |
| test | oracle_all | 1.0000 | 1.0000 | 1.4074 | 2616453.0000 | 0.1190 |
| test | recent | 0.0000 | 0.0000 | 1.0000 | 2616453.0000 | 0.1187 |
| test | uniform | 0.5556 | 0.3519 | 1.0000 | 2616453.0000 | 0.1186 |
| test | wrong | 0.0000 | 0.0000 | 1.0000 | 2616453.0000 | 0.1184 |
| test | empty | 0.0000 | 0.0000 | 0.0000 | 2616453.0000 | 0.1185 |
| test | competing_full | 1.0000 | 1.0000 | 4.0000 | 2884678.7407 | 0.0000 |
| test | competing_memory_full | 1.0000 | 1.0000 | 4.0000 | 2884678.7407 | 0.1230 |
| test | competing_retrieval | 1.0000 | 0.7963 | 1.0000 | 2884678.7407 | 0.1228 |
| test | competing_oracle | 1.0000 | 0.7963 | 1.0000 | 2884678.7407 | 0.1226 |
| test | competing_oracle_all | 1.0000 | 1.0000 | 1.4074 | 2884678.7407 | 0.1226 |
| test | competing_recent | 0.0000 | 0.0000 | 1.0000 | 2884678.7407 | 0.1229 |
| test | competing_uniform | 0.5556 | 0.3519 | 1.0000 | 2884678.7407 | 0.1223 |
| test | competing_wrong | 0.0000 | 0.0000 | 1.0000 | 2884678.7407 | 0.1223 |
| test | competing_empty | 0.0000 | 0.0000 | 0.0000 | 2884678.7407 | 0.1232 |
| train | full | 1.0000 | 1.0000 | 4.0000 | 2621378.9722 | 0.0000 |
| val | full | 1.0000 | 1.0000 | 4.0000 | 2548028.1600 | 0.0000 |
| val | memory_full | 1.0000 | 1.0000 | 4.0000 | 2548028.1600 | 0.1152 |
| val | retrieval | 1.0000 | 0.9700 | 1.0000 | 2548028.1600 | 0.1148 |
| val | oracle | 1.0000 | 0.9700 | 1.0000 | 2548028.1600 | 0.1152 |
| val | oracle_all | 1.0000 | 1.0000 | 1.0600 | 2548028.1600 | 0.1160 |
| val | recent | 0.0000 | 0.0000 | 1.0000 | 2548028.1600 | 0.1158 |
| val | uniform | 0.3600 | 0.3300 | 1.0000 | 2548028.1600 | 0.1161 |
| val | wrong | 0.0000 | 0.0000 | 1.0000 | 2548028.1600 | 0.1161 |
| val | empty | 0.0000 | 0.0000 | 0.0000 | 2548028.1600 | 0.1158 |
| val | episode_capacity2 | 0.4000 | 0.3800 | 0.4000 | 1261544.9800 | 0.0574 |
| val | recent_capacity2 | 0.2800 | 0.2800 | 0.2800 | 1257409.0400 | 0.0577 |
| val | reservoir_capacity2 | 0.5800 | 0.5500 | 0.5800 | 1367167.7000 | 0.0587 |

| Split | Comparison | Model | Accuracy change | Corrected | Regressed | Source-group CI95 |
|---|---|---|---:|---:|---:|---|
| test | memory_full_minus_full | base | -0.0741 | 0 | 2 | [-0.20026315789473684, 0.0] |
| test | memory_full_minus_full | tuned | -0.0370 | 0 | 1 | [-0.0816751700680272, 0.0] |
| test | retrieval_minus_uniform | base | 0.1111 | 5 | 2 | [-0.05, 0.3333333333333333] |
| test | retrieval_minus_uniform | tuned | 0.1111 | 5 | 2 | [-0.05, 0.3333333333333333] |
| test | retrieval_minus_recent | base | 0.3333 | 10 | 1 | [0.10526315789473684, 0.5] |
| test | retrieval_minus_recent | tuned | 0.3333 | 10 | 1 | [0.10526315789473684, 0.5] |
| test | oracle_all_minus_oracle | base | 0.0370 | 1 | 0 | [0.0, 0.17647058823529413] |
| test | oracle_all_minus_oracle | tuned | 0.0370 | 1 | 0 | [0.0, 0.17647058823529413] |
| test | oracle_minus_retrieval | base | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | oracle_minus_retrieval | tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | retrieval_minus_wrong | base | 0.2593 | 9 | 2 | [0.09523809523809523, 0.45011363636363577] |
| test | retrieval_minus_wrong | tuned | 0.2593 | 9 | 2 | [0.09523809523809523, 0.45011363636363577] |
| test | retrieval_minus_empty | base | 0.2963 | 10 | 2 | [0.07130952380952381, 0.55] |
| test | retrieval_minus_empty | tuned | 0.3333 | 11 | 2 | [0.11764705882352941, 0.5911363636363625] |
| test | full_minus_competing_full | base | 0.0000 | 1 | 1 | [-0.16666666666666666, 0.07692307692307693] |
| test | full_minus_competing_full | tuned | 0.0000 | 1 | 1 | [-0.16666666666666666, 0.07692307692307693] |
| test | memory_full_minus_competing_memory_full | base | -0.0370 | 0 | 1 | [-0.18203463203463202, 0.0] |
| test | memory_full_minus_competing_memory_full | tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | retrieval_minus_competing_retrieval | base | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | retrieval_minus_competing_retrieval | tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | oracle_minus_competing_oracle | base | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | oracle_minus_competing_oracle | tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | oracle_all_minus_competing_oracle_all | base | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | oracle_all_minus_competing_oracle_all | tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | recent_minus_competing_recent | base | 0.0370 | 6 | 5 | [-0.2, 0.5602857142857128] |
| test | recent_minus_competing_recent | tuned | 0.0370 | 6 | 5 | [-0.2, 0.5602857142857128] |
| test | uniform_minus_competing_uniform | base | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | uniform_minus_competing_uniform | tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | wrong_minus_competing_wrong | base | 0.0741 | 2 | 0 | [0.0, 0.25] |
| test | wrong_minus_competing_wrong | tuned | 0.0741 | 2 | 0 | [0.0, 0.25] |
| test | empty_minus_competing_empty | base | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | empty_minus_competing_empty | tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | full: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | memory_full: tuned minus base | base→tuned | 0.0370 | 1 | 0 | [0.0, 0.18203463203463086] |
| test | retrieval: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | oracle: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | oracle_all: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | recent: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | uniform: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | wrong: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | empty: tuned minus base | base→tuned | -0.0370 | 0 | 1 | [-0.14285714285714285, 0.0] |
| test | competing_full: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | competing_memory_full: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | competing_retrieval: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | competing_oracle: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | competing_oracle_all: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | competing_recent: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | competing_uniform: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | competing_wrong: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| test | competing_empty: tuned minus base | base→tuned | -0.0370 | 0 | 1 | [-0.14285714285714285, 0.0] |
| train | full: tuned minus base | base→tuned | -0.0093 | 0 | 1 | [-0.01764705882352941, 0.0] |
| val | memory_full_minus_full | base | -0.0400 | 0 | 2 | [-0.10204081632653061, 0.0] |
| val | memory_full_minus_full | tuned | -0.0400 | 1 | 3 | [-0.12244897959183673, 0.03707264957264938] |
| val | retrieval_minus_uniform | base | 0.2200 | 13 | 2 | [0.08, 0.36] |
| val | retrieval_minus_uniform | tuned | 0.2200 | 13 | 2 | [0.08, 0.36] |
| val | retrieval_minus_recent | base | 0.5800 | 30 | 1 | [0.44, 0.72] |
| val | retrieval_minus_recent | tuned | 0.5800 | 30 | 1 | [0.44, 0.72] |
| val | oracle_all_minus_oracle | base | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | oracle_all_minus_oracle | tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | oracle_minus_retrieval | base | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | oracle_minus_retrieval | tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | retrieval_minus_wrong | base | 0.3800 | 21 | 2 | [0.22448979591836735, 0.5306122448979592] |
| val | retrieval_minus_wrong | tuned | 0.3800 | 21 | 2 | [0.22, 0.5306122448979592] |
| val | retrieval_minus_empty | base | 0.4000 | 22 | 2 | [0.23988235294117646, 0.5510204081632653] |
| val | retrieval_minus_empty | tuned | 0.4000 | 22 | 2 | [0.23988235294117646, 0.5510204081632653] |
| val | full: tuned minus base | base→tuned | 0.0200 | 1 | 0 | [0.0, 0.061224489795918366] |
| val | memory_full: tuned minus base | base→tuned | 0.0200 | 1 | 0 | [0.0, 0.061224489795918366] |
| val | retrieval: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | oracle: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | oracle_all: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | recent: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | uniform: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | wrong: tuned minus base | base→tuned | 0.0000 | 1 | 1 | [-0.06, 0.06] |
| val | empty: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | episode_capacity2: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | recent_capacity2: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |
| val | reservoir_capacity2: tuned minus base | base→tuned | 0.0000 | 0 | 0 | [0.0, 0.0] |

- Labels were AI-reviewed; no human validation is claimed.
- Old test results are diagnostic and do not establish generalization.
- Source-group intervals depend on available grouping; hidden overlap may remain.
- Coverage measures retained evidence availability, not whether the reader used it.
- Store bytes exclude model weights; scoring and retrieval latency are reported separately.
