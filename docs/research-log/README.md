# EchoMatch research diary

This directory contains the **chronological research narrative**: our motivation, the questions we investigated, the experiments and controls, the evidence, and the conclusions (including inconclusive ones). Daily logs belong on `master` so a single entry can cover multiple experimental branches. Implementation-specific configurations, Slurm records and reproducibility commands belong in the linked branch protocols.

| Date | Research entry | Main question | Related experiment branches |
| --- | --- | --- | --- |
| 21 Nov 2026 | [Day 01 — EchoMatch overlap weighting on PARTIALSMAL](2026-11-21-day-01-echomatch-overlap.md) | Can predicted overlap help an already trained FM solver? | [Inference experiment](https://github.com/angelhiguera/DeepShapeMatchingKit/tree/experiment/predicted-overlap-fmap) |
| 22–23 Nov 2026 | [Day 02 — Can predicted overlap help EchoMatch learn better correspondences?](2026-11-22-day-02-echomatch-overlap-finetuning.md) | Does using predicted overlap during training help, and must the predictor remain trainable? | [Joint fine-tune](https://github.com/angelhiguera/DeepShapeMatchingKit/tree/experiment/dino-overlap-fm-finetune) · [Frozen predictor](https://github.com/angelhiguera/DeepShapeMatchingKit/tree/experiment/dino-frozen-overlap) |

The [Day 02 log](2026-11-22-day-02-echomatch-overlap-finetuning.md) links to the two separate protocols, with the detailed runs and comparisons. The log is the starting point for understanding the research; the protocols are the reference for reproducing it.
