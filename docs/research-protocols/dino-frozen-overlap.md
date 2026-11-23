# DINO fine-tuning with a frozen overlap predictor

**Question.** Can a fixed set of predicted overlap probabilities help
EchoMatch learn descriptors for correspondence? The earlier joint
fine-tuning let the functional-map (FM) loss reach the overlap predictor.
This experiment supplies the overlap probabilities from the published
model throughout fine-tuning.

## Model and supervision

- The **frozen predictor** uses the published checkpoint's first
  DiffusionNet, learned similarity temperature and overlap DiffusionNet.
  It computes the round trip and predicts overlap for each pair in
  evaluation mode, with dropout disabled and gradients off. It has
  its own copy of those parameters, independent of the trainable extractor.
- The **matching extractor** is a second copy of the published first
  DiffusionNet. Its descriptors feed the FM solver and the original
  InfoNCE. On Y, the solver receives `p_y * F_y`, with the frozen
  probabilities broadcast over the descriptor channels. In the optional
  unmasked comparison, the FM solver receives `F_y` directly.
- The matching extractor is the only part optimized. FM and InfoNCE retain
  their original weights and BCE is disabled. Evaluation loads the saved
  matching extractor and the separate published overlap checkpoint.
  Overlap logits and probabilities come from the frozen predictor
  in both training and evaluation.

| Mode | Weighting of Y in the FM solver during training |
| --- | --- |
| `frozen` | Soft probabilities from the frozen published predictor |
| `frozen-unmasked` | Unit weights, with the same optimizer and losses as `frozen` |

Both use the **304 official train maps**, the same DINO checkpoint, a fresh
Adam at `1e-4`, seed 1234 and **10 epochs** (3,040 configured batches).
The **100 official test maps** are evaluated after training using
`point`, unmasked `fmap`, `fmap_predicted` with the frozen probabilities,
and mask metrics. Use `point` to assess the descriptor quality directly.
The reported mIoU and balanced accuracy belong to the fixed predictor.
They should reproduce the published values when the inputs and environment
match.

The published checkpoint gives the initial reference. The matched unmasked
comparison isolates the contribution of the frozen probabilities during
training. Comparisons against the joint run also change the predictor's
status and BCE supervision. Those comparisons cannot isolate their effects.
An improved overlap predictor can be tested later with the same matching
protocol, reporting both mask quality and correspondence error.

The PyTorch smoke check used the published weights and synthetic operators.
The frozen model reproduced the initial overlap probabilities and matches.
After a student optimizer step, the overlap probabilities stayed identical
and the student extractor changed. The FM loss alone gave finite nonzero
gradients to the student. Real-data training and evaluation are pending.

## Marvin launcher

Use a dedicated checkout when the GT control is queued or running. Slurm
reads the Python files from the checkout at job start, so switching branches
in the submission directory can change which code the job uses. From the
original repository, create a sibling checkout at the frozen experiment's
revision and submit from there:

```bash
git fetch origin
git worktree add --detach "$HOME/DeepShapeMatchingKit-frozen" \
  origin/experiment/dino-frozen-overlap
cd "$HOME/DeepShapeMatchingKit-frozen"
```

The sibling checkout uses the same `../data/PARTIALSMAL` tree. Use the
original `experiment/dino-overlap-fm-finetune` checkout for the matched
GT control. [GT control instructions](https://github.com/angelhiguera/DeepShapeMatchingKit/blob/48071698d2deb8c6387f9079da01b841fa7d4c8f/docs/research-protocols/dino-overlap-fm-finetune.md#run-the-matched-50-epoch-gt-control)

```bash
scratch=/lustre/scratch/data/s23ahigu_hpc-echomatch
export TORCH_HOME="$scratch/torch-cache"
train_job=$(sbatch --parsable --account=ag_ifi_blum --time=02:00:00 \
  --job-name=echo-dino-frozen --output="$scratch/echo-dino-frozen-%j.out" \
  train_partialsmal_dino_overlap.sbatch frozen) &&
eval_job=$(sbatch --parsable --account=ag_ifi_blum --time=00:20:00 \
  --dependency="afterok:${train_job%%;*}" \
  --job-name=echo-dino-evaluate-frozen --output="$scratch/echo-dino-evaluate-frozen-%j.out" \
  train_partialsmal_dino_overlap.sbatch evaluate-frozen) &&
printf 'Training: %s\nEvaluation: %s\n' "$train_job" "$eval_job"
```

To run the matched comparison, use `frozen-unmasked` and
`evaluate-frozen-unmasked` with distinct job names and logs. The two
configurations already use separate checkpoint and result directories.
