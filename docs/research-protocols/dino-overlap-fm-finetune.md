# DINO fine-tuning with predicted overlap in the FM solver

**Technical protocol for the joint predicted-overlap experiment.** For the motivation, the two research ideas, and the comparative conclusions, see the [Day 02 research log on master](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/master/docs/research-log/2026-11-22-day-02-echomatch-overlap-finetuning.md). This file records the differentiable model path, training configurations, matched GT control, logs and reproducibility details.

## Model and gradient path

This experiment starts from the published DINO EchoMatch checkpoint used in
[Day 01](../research-log/2026-11-21-day-01-echomatch-overlap.md). The published
forward pass derives the overlap head's input from soft correspondence
round trips built with the shared descriptors. The head processes those
signals and outputs an overlap probability per vertex. The original
descriptors also feed the FM solver and the InfoNCE objective.

The experiment asks whether conditioning FM training on the predicted overlap
can make the learned descriptors more useful for correspondence. In
`predicted_y`, the solver receives `p_y * F_y`, with `F_y` shaped
`[1, M, D]` and `p_y` reshaped to `[1, M, 1]`. X enters unchanged.
The probabilities participate in autograd and feed the same spectral
projection and linear solve as the GT-weighted descriptors. The FM loss
can therefore reach the overlap head and the shared descriptor extractor.
BCE still supervises overlap on both shapes, and InfoNCE still uses the
original descriptors.

## Protocol

| Checkpoint | FM weighting in training | Initialization |
| --- | --- | --- |
| Published | Initial reference for evaluation | Published DINO checkpoint |
| `predicted` | Soft predicted overlap on Y from the first update. | Same published weights, fresh Adam and counters |
| `gt` (matched control) | GT overlap on Y, the original `f2_only` route. | Same published weights, fresh Adam and counters |

Use the **official PARTIALSMAL splits**. Fine-tuning uses the full official
training set of 304 directional maps. The configuration uses seed 1234
for reproducibility, one map per update, and Adam at `1e-4`. The initial
pilot ran for 10 epochs (3,040 updates); subsequent predicted-weight runs
reached 50 and 100 epochs, and the matched GT control reached 50 epochs. Keep the original loss weights and save a
numbered recovery checkpoint every 304 updates. Evaluate the weights from
`final.pth`, exported after the full budget. The strict checkpoint check
prevents silent random initialization if the published file is absent.
The launcher logs the map-set and checkpoint SHA-256 values.

A direct switch from GT masks to soft predictions is the first comparison.
The pretrained checkpoint makes that comparison feasible from step zero.
Day 01's mixed frozen-inference results still leave training stability open.
A GT-to-predicted schedule is a separate follow-up if the direct switch
leads to unstable optimization. Comparing such schedules would require
development data apart from the official test.

There is no training-time validation or checkpoint selection in this pilot.
The evaluation uses the published checkpoint and the completed
predicted-mask fine-tune on the same 100 official test maps. This measures
the change from published weights to the fine-tuned model as a whole.
The later matched GT fine-tune at 50 epochs partly separates the effect
of the predicted mask from the effect of extra updates (see run records below). The upstream
DINO training YAML lists the official test data
for validation. The published checkpoint's historical selection protocol
remains unverified. The new fine-tuning stage has no access to the official
test through its training configuration.

Compare `point`, unmasked `fmap`, `fmap_predicted` and overlap mIoU on
the published and fine-tuned checkpoints. Better `point` matching
would support a gain in the descriptors. A gain specific to
`fmap_predicted` would suggest a benefit from how the mask and solver
interact. Track FM loss, BCE and InfoNCE during training.

## Marvin launcher

Run from the repository root on Marvin. The launcher defaults to the official
dataset root `../data/PARTIALSMAL`. To use another installation, pass its
PARTIALSMAL root as the second argument. The `check` mode builds the
published model, runs two real training samples through both strategies, and
checks the FM-loss gradient into the overlap head. It performs zero optimizer
updates.

```bash
export TORCH_HOME=/lustre/scratch/data/s23ahigu_hpc-echomatch/torch-cache
sbatch --account=ag_ifi_blum train_partialsmal_dino_overlap.sbatch check
```

For a cold DINO cache, `cache` traverses the official training set on an
allocated GPU. DINO features cached under the older scratch split live in a
different directory. Use a primed `TORCH_HOME` if the compute node has
restricted network access. After the gradient check, run `predicted`.
Once it completes, `evaluate` compares its `final.pth` with the published
checkpoint, writing a separate output directory for each. The optional
`gt` and `evaluate-gt` modes can add the matched GT control later.

### Run 50 epochs from the published checkpoint

`predicted50` loads only the published DINO network weights. It starts at
epoch 1 with a fresh optimizer and scheduler and runs through epoch 50
(15,200 batches). It writes its checkpoints to the separate
`echo_match_psmal_dino_overlap_finetune_official_predicted50` run directory.

From the repository root with the experiment branch up to date, schedule
training and the fixed four-checkpoint evaluation:

```bash
scratch=/lustre/scratch/data/s23ahigu_hpc-echomatch
export TORCH_HOME="$scratch/torch-cache"
train_job=$(sbatch --parsable --account=ag_ifi_blum --time=06:00:00 \
  --job-name=echo-dino-predicted50 --output="$scratch/echo-dino-predicted50-%j.out" \
  train_partialsmal_dino_overlap.sbatch predicted50) &&
eval_job=$(sbatch --parsable --account=ag_ifi_blum --time=00:20:00 \
  --dependency="afterok:${train_job%%;*}" \
  --job-name=echo-dino-evaluate50 --output="$scratch/echo-dino-evaluate50-%j.out" \
  train_partialsmal_dino_overlap.sbatch evaluate50) &&
printf 'Training: %s\nEvaluation: %s\n' "$train_job" "$eval_job"
```

The evaluation prints blocks for the published checkpoint and epochs 10,
20 and 50 of the new run. Each has its own results directory. Report the
complete comparison on official test after training.

### Run the matched 50-epoch GT control

`gt50` uses the same published checkpoint, fresh optimizer and scheduler,
seed, budget, losses, and 304 official training maps as `predicted50`. The
training difference is the FM weighting on Y. The GT control uses the
original `f2_only` strategy. Both runs use the same test configuration to
measure `point`, `fmap`, `fmap_predicted`, and overlap metrics.

From the repository root on Marvin:

```bash
scratch=/lustre/scratch/data/s23ahigu_hpc-echomatch
export TORCH_HOME="$scratch/torch-cache"
train_job=$(sbatch --parsable --account=ag_ifi_blum --time=06:00:00 \
  --job-name=echo-dino-gt50 --output="$scratch/echo-dino-gt50-%j.out" \
  train_partialsmal_dino_overlap.sbatch gt50) &&
eval_job=$(sbatch --parsable --account=ag_ifi_blum --time=00:20:00 \
  --dependency="afterok:${train_job%%;*}" \
  --job-name=echo-dino-evaluate-gt50 --output="$scratch/echo-dino-evaluate-gt50-%j.out" \
  train_partialsmal_dino_overlap.sbatch evaluate-gt50) &&
printf 'Training: %s\nEvaluation: %s\n' "$train_job" "$eval_job"
```

The control saves weights under `echo_match_psmal_dino_overlap_finetune_official_gt50`.
The evaluation writes the three checkpoint results under separate
`_official_test_gt_curve50_*` directories. Compare all checkpoints at epochs
10, 20, and 50 against their predicted-overlap counterparts. The published
reference is already recorded by `evaluate50`.

#### Motivation and submission on 23 November 2026

The 50-epoch predicted run lowered `point` error from 0.0493 to 0.0460.
It combined extra optimization with a change to the FM training weights.
The matched GT control asks how much the original training rule improves
the same checkpoint over the same budget. A similar gain with GT would
show that extra fine-tuning under the original rule can account for the
observed improvement. A lower error with predicted weighting would
support a contribution from that training route under the matched setup.

Both control jobs were submitted from the detached
`DeepShapeMatchingKit-gt50` checkout at [`4807169`](https://github.com/angelhiguera/DeepShapeMatchingKit/commit/48071698d2deb8c6387f9079da01b841fa7d4c8f).

| Job | Mode | Output log |
| --- | --- | --- |
| `29299028` | `gt50`, 50 epochs on official train | `echo-dino-gt50-29299028.out` |
| `29299029` | `evaluate-gt50`, official test after successful training | `echo-dino-evaluate-gt50-29299029.out` |

The supplied evaluation log contains all three metric blocks. Against
predicted training job `29282882` and evaluation job `29282883`, GT has
lower `point` error at epochs 10 and 20, and predicted weighting has
lower `point` error at epoch 50. The original rule substantially improves
the published descriptor matching by itself. [Complete matched results](#matched-gt-control-results)


## Results and run provenance (22–23 November 2026)

These detailed records were moved from the Day 02 research log so that the [canonical daily narrative](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/master/docs/research-log/2026-11-22-day-02-echomatch-overlap-finetuning.md) can compare the two research directions without mixing in run-specific details.

### Ten-epoch pilot: gradients, jobs and scores

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

#### Gradient check

Measured the gradient of **FM loss alone** with respect to overlap-head
parameters on the same samples and initial weights.

| Direction | GT weighting | Predicted weighting |
| --- | --- | --- |
| `cuts_14_cow_06` → `cuts_6_fox_03` | No gradient into the head | Norm 2.10025, 31/31 parameter tensors connected |
| `cuts_1_dog_06` → `cuts_18_fox_02` | No gradient into the head | Norm 6.40092, 31/31 parameter tensors connected |

Both predicted-path norms were finite and nonzero. This establishes a
training signal on these samples. The check does not measure how much it
improves correspondence or whether the signal remains stable over longer runs.

#### Official test results

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

### Predicted-overlap trajectory through 50 epochs

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

### Matched GT control results

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

### Predicted-overlap trajectory through 100 epochs

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
