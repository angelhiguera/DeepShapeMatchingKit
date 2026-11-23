# DINO fine-tuning with predicted overlap in the FM solver

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
| `gt` (optional) | GT overlap on Y, the original `f2_only` route. | Same published weights, fresh Adam and counters |

Use the **official PARTIALSMAL splits**. Fine-tuning uses the full official
training set of 304 directional maps. The configuration uses seed 1234
for reproducibility, one map per update, Adam at `1e-4`, and a fixed budget
of 10 epochs (3,040 updates). Keep the original loss weights and save a
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
An optional GT fine-tune at the same budget would help separate the effect
of the predicted mask from the effect of the extra updates. The upstream
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
