# Day 02 — Fine-tuning EchoMatch through predicted overlap

**22 November 2026 · Published DINO initialization · Official PARTIALSMAL splits**

## Research questions

[Day 01](2026-11-21-day-01-echomatch-overlap.md) tested predicted overlap as an
inference-time intervention with frozen descriptors. Here we ask whether
letting the functional-map (FM) loss use the predicted overlap during
training can help EchoMatch learn better descriptors.

1. **Can the FM loss reach the overlap predictor?** The baseline gives the
   solver the GT overlap mask on Y during training. Replacing it with the
   predicted probabilities should open a gradient path from the FM loss
   through the solver to the overlap head.
2. **Does the adjusted model learn descriptors that match better?** The
   primary readout is direct descriptor matching (`point`). Unmasked
   `fmap` and `fmap_predicted` distinguish improvements visible in the
   descriptors from gains that appear when the solver uses the weights.
3. **Does mask quality change?** Report overlap mIoU and balanced accuracy
   alongside geodesic error. A similar global IoU can conceal changes
   in where the predictor makes errors.

## Completed experiment

The model started from the published DINO checkpoint and used continuous
predicted probabilities on Y from the first update. This weighting applies
to the FM solver. BCE and both InfoNCE terms retain their original
definitions. The pilot used seed 1234, Adam at `1e-4`, and 10 epochs over
the **304 directional maps in official train** for 3,040 configured updates.
The evaluation compared the published and final fine-tuned checkpoints
on the **100 official test maps**. The test results were recorded after the
fixed-budget training run. [Protocol](../research-protocols/dino-overlap-fm-finetune.md)

| Marvin job | Outcome |
| --- | --- |
| `29280263` | Gradient check completed in 1 min 13 s on two real train pairs, with zero optimizer updates. |
| `29280264` | Predicted-overlap fine-tuning completed in 53 min 21 s. |
| `29280265` | Evaluation of the two checkpoints completed in 1 min 58 s. |

All three jobs reported exit code `0:0`. The record uses the supplied
Slurm status and evaluation log excerpts. The launcher version is
[`f1a19f5`](https://github.com/angelhiguera/DeepShapeMatchingKit/commit/f1a19f5b7875dabab2fa44752dd569eefed50b49).
Archive the full job logs with the checkpoint and map-set digests they report
to confirm the exact run provenance.

### Gradient check

Measured the gradient of **FM loss alone** with respect to overlap-head
parameters on the same samples and initial weights.

| Direction | GT weighting | Predicted weighting |
| --- | --- | --- |
| `cuts_14_cow_06` → `cuts_6_fox_03` | No gradient into the head | Norm 2.10025, 31/31 parameter tensors connected |
| `cuts_1_dog_06` → `cuts_18_fox_02` | No gradient into the head | Norm 6.40092, 31/31 parameter tensors connected |

Both predicted-path norms were finite and nonzero. This establishes a
training signal on these samples. The check does not measure how much it
improves correspondence or whether the signal remains stable over longer runs.

### Official test results

| Metric | Published DINO | After 10 epochs with predicted overlap | Change |
| --- | ---: | ---: | ---: |
| `point` mean geodesic error ↓ | 0.0493 | **0.0463** | −6.1% relative |
| `fmap` without a mask ↓ | 0.0771 | **0.0758** | −1.7% relative |
| `fmap_predicted` ↓ | 0.0757 | **0.0746** | −1.5% relative |
| Mask mIoU ↑ | 84.75% | 84.78% | +0.03 percentage points |
| Balanced mask accuracy ↑ | 92.75% | 92.83% | +0.08 percentage points |

The mask values were identical for both shape roles at the precision logged.
Relative changes above use the rounded job output.

The largest observed gain was in direct descriptor matching. Unmasked FM
also improved, while the global mask metrics changed little. The incremental
effect of using the mask at inference was similar for both checkpoints
(−0.0014 published and −0.0012 after fine-tuning). This supports a gain in
descriptor matching after the full fine-tuning intervention. How much of
that gain comes from predicted weighting and how much comes from the extra
training remains open. The recorded results are from one fine-tuning run
and aggregate test logs.

## Follow-up: 50 epochs from the published checkpoint

**Question.** Does the descriptor-matching gain persist or grow with more
fine-tuning?

Training job `29282882` ran 50 epochs (15,200 configured batches) in
41 min 05 s. It started at epoch 0 and iteration 0 with the published DINO
weights. The logged SHA-256 matches the Day 01 checkpoint digest
`acbe75fd2091980fb9628f4458b118287af5639e11f05023c1af42a69849fe59`.
Evaluation job `29282883` completed in 2 min 39 s. Both reported exit code
`0:0`. The reported results are from
`echo-dino-evaluate50-29282883.out`. [Launcher revision](https://github.com/angelhiguera/DeepShapeMatchingKit/commit/8d0972e59d6d725aa202f40f278fae2dd048833a) · [Marvin commands](../research-protocols/dino-overlap-fm-finetune.md#run-50-epochs-from-the-published-checkpoint)

The three fine-tuned checkpoints come from one run, with the same losses,
learning rate and official training split as the 10-epoch pilot. The
endpoint and evaluation points were fixed before this run.

| Checkpoint | `point` error ↓ | `fmap` error ↓ | `fmap_predicted` error ↓ | Mask mIoU ↑ | Balanced accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published | 0.0493 | 0.0771 | 0.0757 | 84.75% | 92.75% |
| Epoch 10 | 0.0463 | 0.0758 | 0.0746 | 84.78% | 92.83% |
| Epoch 20 | 0.0473 | 0.0770 | 0.0748 | **85.40%** | **93.03%** |
| Epoch 50 | **0.0460** | **0.0750** | **0.0738** | 84.89% | 92.91% |

Epoch 10 matches the pilot on all five reported metrics at log precision.
Epoch 50 has the lowest error among the measured checkpoints for each
matching method. For `point`, the relative reduction is **6.7% versus the
published checkpoint** and **0.65% versus epoch 10**. Most of the observed
descriptor gain therefore appears by epoch 10. All three matching errors
rise at epoch 20 relative to epoch 10 before improving at epoch 50.

Mask IoU and balanced accuracy peak at epoch 20, where the matching errors
are worse than at epochs 10 and 50. Overall mask agreement and
correspondence quality move differently on this trajectory. The
improvements are descriptive. This is one exploratory trajectory motivated
by the pilot's test result, and the logs provide rounded aggregates.
The contribution of predicted weighting remains entangled with the
effect of extra training.

## Matched GT and frozen-predictor results on 23 November

### Matched GT control

The GT control changed only the overlap weights supplied to the FM solver
during training. It used the published initialization and the same data,
seed and budget as the 50-epoch predicted run. The first, second and third
metric blocks in job `29299029` correspond to epochs 10, 20 and 50 in
that order, as defined by the submitted launcher. The GT label refers to
training. At test time `fmap_predicted` uses the model's own predicted
probabilities for each checkpoint. [Fixed evaluation order](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/48071698d2deb8c6387f9079da01b841fa7d4c8f/train_partialsmal_dino_overlap.sbatch#L137)

| Training weights | Epoch | `point` ↓ | `fmap` ↓ | `fmap_predicted` ↓ | Mask mIoU ↑ | Balanced accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GT | 10 | 0.0456 | 0.0760 | 0.0739 | 85.28% | 93.10% |
| Predicted | 10 | 0.0463 | 0.0758 | 0.0746 | 84.78% | 92.83% |
| GT | 20 | 0.0467 | 0.0762 | 0.0741 | 85.28% | 93.03% |
| Predicted | 20 | 0.0473 | 0.0770 | 0.0748 | 85.40% | 93.03% |
| GT | 50 | 0.0466 | 0.0751 | 0.0741 | 84.08% | 92.48% |
| Predicted | 50 | 0.0460 | 0.0750 | 0.0738 | 84.89% | 92.91% |

The original GT training rule is sufficient to improve `point` over
the published 0.0493. At epochs 10 and 20 the GT control has lower
`point` and `fmap_predicted` error. At epoch 50 the predicted run has
slightly lower error on all three matching readouts. Through epoch 50,
the lowest `point` error is GT at epoch 10 (0.0456),
and the lowest masked FM error is predicted at epoch 50 (0.0738 versus
GT's 0.0739 at epoch 10). This supports a benefit from fine-tuning and
shows a difference in the trajectories under the two training rules.
It gives limited evidence for a general advantage of predicted weights.
The comparisons use one seed and rounded test aggregates.

### Frozen predictor: matched control and training duration

The frozen run trains the matching extractor with FM and InfoNCE and uses
the fixed published predictor for the FM weighting. The official test
log for `29299031` shows `point` improving from **0.0493 to 0.0468**
(5.1%). Unmasked FM improves from 0.0771 to 0.0767, while masked
FM moves from 0.0757 to 0.0767. For the fine-tuned extractor, the
two FM errors round to the same value. Mask mIoU remains 84.75% and
balanced accuracy remains 92.75%, as expected for the fixed predictor.
The matched `frozen-unmasked` run (training `29315145`, evaluation
`29315146`) used the same fixed predictor and FM/InfoNCE losses. At
10 epochs, training with its weights gave nearly the same `point` as
training without them (0.0468 versus 0.0469), and lower unmasked FM
error (0.0767 versus 0.0800) and masked FM error (0.0767 versus 0.0785).

A fresh `frozen` run to 50 epochs (training `29316280`, evaluation
`29317866`) kept `point` in a narrow range of 0.0466–0.0468.
From epochs 10 to 50, unmasked FM error fell from 0.0767 to 0.0759,
and masked FM error fell from 0.0767 to 0.0757. The mIoU stayed at
84.75% throughout. The fixed weights were more useful for the FM readouts
than for direct matching in these comparisons. [Full frozen results](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/348c4e1ca8c5ea981012e417e8df08ebf7249354/docs/research-protocols/dino-frozen-overlap.md#official-test-results)

### Joint predicted weighting through 100 epochs

Job `29315653` was submitted as a fresh 100-epoch fine-tune from the
published DINO weights, with predicted weighting in the training solver.
It completed in 1 h 22 min 24 s. Job `29318393` evaluated `15200.pth`
and `final.pth` from that same run on the official test. Source logs:
`echo-dino-predicted100-29315653.out` and
`echo-dino-compare-50-100-29318393.out`.

| Epoch | `point` ↓ | `fmap` ↓ | `fmap_predicted` ↓ | Mask mIoU ↑ | Balanced accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.0460 | 0.0750 | 0.0738 | 84.89% | 92.91% |
| 100 | 0.0460 | **0.0737** | **0.0731** | **85.58%** | **93.18%** |

The epoch-50 snapshot reproduces the earlier 50-epoch run at log
precision. The next 50 epochs improve unmasked and masked FM by 1.7%
and 0.9% relative to epoch 50, and increase mIoU by 0.69 percentage
points. Direct descriptor matching stays at 0.0460 at log precision.
The additional benefit of applying the mask at inference shrinks from
0.0012 to 0.0006 while both FM errors improve. The 100-epoch comparison
describes the joint run's trajectory. The GT control covers up to 50
epochs.

### Conclusions across the fine-tuning experiments

Fine-tuning from the published DINO checkpoint improves direct matching
under the original GT rule and under predicted weighting. At matched
budgets, the lead in `point` changes with the checkpoint. With the
predictor fixed, the matched weighting comparison shows a clearer
benefit for FM than for direct matching. Longer training continued to
improve FM in both the frozen and joint runs, with little additional
change in `point`. The evidence supports a modest FM benefit under
the tested fine-tuning setup. An advantage for learning direct matching
descriptors through predicted weighting remains uncertain. These results
use one seed and rounded official-test aggregates.

## Experiment branches and status

| Branch | Scope |
| --- | --- |
| `experiment/dino-overlap-fm-finetune` | Joint predicted weighting up to 100 epochs and the matched GT control up to 50. The planned evaluations are recorded above. |
| `experiment/dino-frozen-overlap` | Fixed-teacher weighting, the matched `frozen-unmasked` control at 10 epochs, and the weighted trajectory up to 50. [Protocol](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/348c4e1ca8c5ea981012e417e8df08ebf7249354/docs/research-protocols/dino-frozen-overlap.md) |
| `experiment/predicted-overlap-fmap` | Earlier frozen-checkpoint inference, per-pair and vertex diagnostics, and the viewer. [Experiment history](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/4816bdd944632dd6dea4836e3d0268d6e21606c3/docs/PREDICTED_OVERLAP_HISTORY.md) |

## Open experiments

### Test overlap-aware cross-shape InfoNCE separately

**Question.** Do descriptors improve when vertices likely outside the
overlap have less influence as candidate correspondences in cross-shape
InfoNCE?

Hold the FM strategy fixed and compare original InfoNCE with a variant that
adds `log(w_j)` to Y candidate logits, and symmetrically to X candidate
logits in reverse. Keep each GT positive at weight 1 and give other candidates
a positive weight floor based on the predicted overlap. Start with detached
probabilities. This isolates the effect of changing the candidate weights
on descriptor learning. The within-shape InfoNCE term stays as defined
in the baseline. Set the candidate-weight rule before the test comparison.
Use direct `point` matching as the primary metric.

### Investigate the overlap head's intermediate descriptors

Reusing the overlap head's internal representations as inputs to an FM
solver is a separate, deferred idea. It needs a defined feature layer
and a matched descriptor comparison before implementation. This idea
and overlap-aware InfoNCE have no implementation branch as of this entry.
