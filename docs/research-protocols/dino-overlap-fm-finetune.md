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

| Run | FM weighting in training | Initialization |
| --- | --- | --- |
| `baseline` | No training. Evaluate the original checkpoint on the validation fold at step zero. | Published DINO checkpoint |
| `gt` | GT overlap on Y, the original `f2_only` route. | Same published weights, fresh Adam and counters |
| `predicted` | Soft predicted overlap on Y from the first update. | Same published weights, fresh Adam and counters |

This initial pilot uses seed 1234, one directional map per update, 10 epochs
(2,720 updates), and Adam at `1e-4`. Both training arms keep the original
loss weights, evaluate and save every 272 updates, and select their best
checkpoint by validation `point` error. The strict checkpoint check
prevents silent random initialization if the file is absent. The
launcher logs the checkpoint and split-manifest SHA-256 for each mode.

A direct switch from GT masks to soft predictions is the first comparison.
The pretrained checkpoint makes that comparison feasible from step zero.
Day 01's mixed frozen-inference results still leave training stability open.
A GT-to-predicted schedule is a follow-up if the direct switch leads to
unstable optimization or worse results on the development fold.

The seed-1234 split has 272 train and 32 validation directional maps. The
published checkpoint may have seen the validation pairs during its original
training, since they were drawn from the official training data. Use this
fold for fine-tuning development and checkpoint selection. Hold the official
test split for evaluation after decisions are fixed.

Compare `point`, unmasked `fmap`, `fmap_predicted` and overlap mIoU at
step zero and across both fine-tuned checkpoints. Better `point` matching
would support a gain in the descriptors. A gain specific to
`fmap_predicted` would suggest a benefit from how the mask and solver
interact. Track FM loss, BCE and InfoNCE during training, and inspect
validation by direction if aggregates diverge.

## Marvin launcher

Run from the repository root on Marvin. Point the launcher to the existing
research split and use your group Slurm account. The `check` mode builds the
published model, runs two real training samples through both strategies, and
checks the FM-loss gradient into the overlap head. It performs zero optimizer
updates.

```bash
split=/lustre/scratch/data/s23ahigu_hpc-echomatch/partialsmal_split_seed1234
sbatch --account=ag_ifi_blum train_partialsmal_dino_overlap.sbatch check "$split"
```

For a cold DINO cache, `cache` traverses the train and validation folds on
an allocated GPU. The DINO Hub cache defaults to `torch-cache` alongside
the split. Set `TORCH_HOME` to a primed location if the compute node has
restricted network access. Launch `baseline`, `gt` and `predicted` with
the same split to run the comparison.
