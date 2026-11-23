# Day 02 — Can predicted overlap help EchoMatch learn better correspondences?

**22 November 2026 · Follow-up experiments and controls: 23 November 2026 · DINO EchoMatch on PARTIALSMAL**

This is the **research narrative**: why we tried these ideas, what each experiment tested, what the results mean, and what remains unresolved. Reproducibility details, run identifiers, configurations and full evaluation records live in the linked **experiment-specific protocols**. The two ideas are implemented on separate branches.

## From Day 01 to a training hypothesis

On [Day 01](2026-11-21-day-01-echomatch-overlap.md), we asked whether EchoMatch's existing predicted overlap could help functional-map (FM) matching **without retraining the model**. We compared direct nearest-neighbor descriptor matching (`point`), the original unmasked FM (`fmap`), FM weighted by *predicted* overlap (`fmap_predicted`), and an oracle GT-mask variant on the official PARTIALSMAL test set.

Predicted weighting reduced the pooled FM geodesic error from **0.077141 to 0.075736**. It was not a uniform win across individual shape pairs, and direct descriptor matching was still substantially better (**0.049341**). The experiment suggested that the overlap signal could sometimes help the solver, but it **did not tell us whether using that signal during training would improve the learned descriptors**.

There is also an important asymmetry in the original EchoMatch setup: the model **predicts overlap**, but its FM loss during training uses **ground-truth overlap on shape Y**. In ordinary evaluation, the FM solver uses unmasked descriptors. This raised our central question:

> **Can predicted overlap become a useful training signal for correspondence learning, rather than only an inference-time adjustment?**

A gain after training would not automatically answer *why* it happened. It might come from learning better descriptors, from using overlap weights in the solver, from improving the overlap predictor, or simply from giving the pretrained model more optimization steps. We designed two branches and matched controls to separate some of those explanations.

## Idea 1 — Let the FM loss train through predicted overlap

**Research question.** What happens if, instead of supplying GT overlap to the FM solver during training, we weight Y's descriptors by the model's own **continuous predicted overlap probabilities**? Can the resulting FM gradients improve correspondences and the overlap predictor?

**Experiment.** We initialized from the published DINO EchoMatch checkpoint, replaced the GT weighting with predicted probabilities, and kept the original FM, BCE and InfoNCE objectives. Unlike the baseline's GT mask, the soft predicted weights are part of the computation graph: **FM gradients can flow into the overlap head and the shared descriptor extractor**. We checked that gradient path explicitly and fine-tuned for 10, 50 and 100 epochs. To test the effect of *predicted* weighting rather than just extra training, we also ran a **matched 50-epoch GT-weighted fine-tuning control**.

**What we learned.** The gradient check succeeded. Joint fine-tuning improved direct matching and FM errors over the published checkpoint. By 100 epochs, `point` error had decreased from **0.0493 to 0.0460**, and `fmap_predicted` from **0.0757 to 0.0731**. However, the GT control also improved `point`: GT was better at epochs 10 and 20, while predicted weighting was slightly better at epoch 50. Thus, **the improvement relative to the published checkpoint is real for this run, but a consistent advantage attributable specifically to predicted weighting is not established**. Extending joint fine-tuning from 50 to 100 epochs improved FM metrics without a measurable change in `point` at the reported precision.

**Branch and detailed protocol:** [`experiment/dino-overlap-fm-finetune`](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/experiment/dino-overlap-fm-finetune/docs/research-protocols/dino-overlap-fm-finetune.md) — architecture and gradient path, controls, job records, training commands and full results.

## Idea 2 — Can a *frozen* overlap predictor guide descriptor learning?

**Research question.** Does the overlap predictor have to adapt for correspondence learning to improve? Or is the overlap signal from the published checkpoint already useful enough to guide a separate trainable descriptor extractor?

**Experiment.** We kept an independent copy of the published predictor **fully frozen**, including the feature extractor that produces the inputs to its overlap head. Freezing only the final head would not have guaranteed constant predictions as the shared extractor changed. The frozen teacher supplied the same overlap probabilities for a given input pair throughout training, while a student extractor learned under FM and InfoNCE losses; overlap BCE was disabled. We compared **frozen-weighted** and **frozen-unmasked** students at the same 10-epoch budget, then examined the weighted run through 50 epochs.

**What we learned.** At 10 epochs, both students improved `point` similarly (to **0.0468** with weighting and **0.0469** without it). But weighting improved the FM readouts much more clearly: compared with the matched unmasked training, `fmap` fell from **0.0800 to 0.0767** and `fmap_predicted` from **0.0785 to 0.0767**. The predictor's mask mIoU stayed at **84.75%**, as expected. Further weighted training improved FM errors but barely changed direct descriptor matching. **An already-trained, fixed overlap signal can help learn descriptors that work better with the FM solver, without demonstrating a meaningful extra gain in direct nearest-neighbor matching.**

**Branch and detailed protocol:** [`experiment/dino-frozen-overlap`](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/experiment/dino-frozen-overlap/docs/research-protocols/dino-frozen-overlap.md) — frozen teacher/student separation, matched no-weighting control, jobs, commands and full results.

## Results and what the controls tell us

All reported comparisons below use the **304 official training maps**, **100 official test maps** and the published DINO checkpoint as the starting point. Lower correspondence error is better. `point` assesses nearest-neighbor matching directly in descriptor space; `fmap` is unmasked FM; `fmap_predicted` weights overlap in the evaluation solver. The last column measures predicted-overlap mask quality, **not** matching accuracy.

| Training strategy / checkpoint | `point` ↓ | `fmap` ↓ | `fmap_predicted` ↓ | Mask mIoU ↑ |
| --- | ---: | ---: | ---: | ---: |
| Published DINO checkpoint | 0.0493 | 0.0771 | 0.0757 | 84.75% |
| GT overlap fine-tune, epoch 10 | **0.0456** | 0.0760 | 0.0739 | 85.28% |
| Joint predicted overlap, epoch 10 | 0.0463 | 0.0758 | 0.0746 | 84.78% |
| GT overlap fine-tune, epoch 50 | 0.0466 | 0.0751 | 0.0741 | 84.08% |
| Joint predicted overlap, epoch 50 | 0.0460 | 0.0750 | 0.0738 | 84.89% |
| Joint predicted overlap, epoch 100 | 0.0460 | **0.0737** | **0.0731** | **85.58%** |
| Frozen teacher, **no** FM weighting, epoch 10 | 0.0469 | 0.0800 | 0.0785 | 84.75% |
| Frozen teacher, predicted FM weighting, epoch 10 | 0.0468 | 0.0767 | 0.0767 | 84.75% |
| Frozen teacher, predicted FM weighting, epoch 50 | 0.0467 | 0.0759 | 0.0757 | 84.75% |

**Three distinctions matter when interpreting these numbers:**

1. **Fine-tuning itself helps.** The matched GT run improves `point` over the published model, so a published-versus-predicted comparison alone cannot establish a benefit from predicted weighting. The relative lead between GT and predicted weighting changes over the 10–50-epoch trajectory.
2. **Better overlap masks are not the same as better correspondences.** In the joint run, mask mIoU and matching accuracy do not peak at the same epoch. In the frozen run, matching changes even though the overlap predictions are fixed.
3. **FM gains and descriptor gains are different claims.** The frozen weighted-vs-unmasked control supports a benefit to the FM readout, but its `point` difference is just 0.0001 at log precision. Moreover, frozen vs joint fine-tuning is **not** a fully controlled head-to-head test: freezing the teacher also removes BCE and changes which parameters are trainable.

## Conclusions and next questions

The experiments give evidence that **overlap information can be used during training to improve FM correspondence behavior**. They do **not** establish that predicted overlap is always superior to GT overlap, that a better mask IoU reliably produces better correspondences, or that a trainable predictor is necessary for the improvement seen with a frozen teacher.

These are exploratory, single-seed fine-tuning runs. Metrics are rounded official-test aggregates; the matched GT comparison ends at 50 epochs, whereas the joint trajectory reaches 100. We should avoid choosing new methods or checkpoints based only on repeatedly inspecting the official test set: subsequent attribution experiments should have a separate development/validation split.

A useful next direction is to ask whether overlap should influence **descriptor learning itself**, not only the FM solver. For example, overlap-aware cross-shape InfoNCE could downweight non-overlapping candidate negatives while preserving GT positives. Another distinct hypothesis is a **GT-to-predicted weighting schedule** during training, introduced only with a controlled validation protocol. Neither is an outcome of Day 02.

## Code, protocols and reproducibility

- **Joint predictor + descriptor fine-tuning:** [`experiment/dino-overlap-fm-finetune` protocol](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/experiment/dino-overlap-fm-finetune/docs/research-protocols/dino-overlap-fm-finetune.md). Includes the published/GT/predicted comparisons, gradient verification, full run records and Marvin commands.
- **Frozen predictor + trainable descriptor extractor:** [`experiment/dino-frozen-overlap` protocol](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/experiment/dino-frozen-overlap/docs/research-protocols/dino-frozen-overlap.md). Includes the matched unweighted control and the 10–50-epoch fixed-teacher results.
- **Previous experiment:** [Day 01 — inference-time overlap weighting](2026-11-21-day-01-echomatch-overlap.md).

**Documentation convention:** The daily research logs live on `master` and describe the questions, decisions and findings across branches. Each experiment branch owns its implementation and technical protocol. This allows an individual day's research to develop into more than one independent experiment without duplicating its narrative.
