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

## Proposed next experiments

### 1. Extend predicted-overlap fine-tuning to 50 epochs total

**Question.** Does the descriptor-matching gain persist or grow across a
longer adjustment?

Fix the endpoint at **50 epochs** (15,200 batches) in one new run. Use
the same published initialization, losses, learning rate and data as the
pilot. Compare the published weights and the epoch-10, epoch-20 and
epoch-50 checkpoints from this run on the same test maps after training.
Use `point` as the primary result and track both FM variants and mask mIoU
as secondary readouts. Report all four measurements, including any
deterioration. This follow-up is exploratory because it is motivated by the
10-epoch pilot's test result.

The `predicted50` launcher loads the published DINO network weights and
starts from epoch 1 with fresh Adam and StepLR state. The learning rate
remains `1e-4` through epoch 50 under `StepLR(step_size=100)`. The
`evaluate50` launcher reads `3040.pth`, `6080.pth` and `final.pth`
from the long run alongside the published checkpoint.
[Marvin commands](../research-protocols/dino-overlap-fm-finetune.md#run-50-epochs-from-the-published-checkpoint)

### 2. Optionally match additional training with GT masks

**Question.** What portion of the gain is specific to letting the FM solver
use predicted overlap, given the same number of extra updates?

Another 10 epochs with the original GT weighting would directly match the
completed pilot. If the 50-epoch result drives the next conclusion, compare
against a GT run at that same budget. Start from the published checkpoint
in both conditions. This control is most valuable if we want to attribute
the gain to predicted weighting. It can wait while we establish whether the
predicted-mask trajectory is worth pursuing.

### 3. Test overlap-aware cross-shape InfoNCE separately

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
