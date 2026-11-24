# PARTIALSMAL DINO matching-only reference

After the advisor meeting, we want a matching-only performance reference
before improving the initial descriptors or changing the overlap predictor,
including potential VLM-based approaches. The question is how well the
current EchoMatch matching stack learns correspondence when trained from
random EchoMatch weights with its existing DINO inputs and without the
overlap prediction branch. The result applies to this architecture,
training budget, and selection protocol.

## Architecture and protocol

- Set `networks.echo_match_net.overlap_predictor: none` to omit the learned
  temperature, soft maps, cycle matrices, second DiffusionNet, and overlap
  BCE. The predictor called by the original configurations receives the two
  shape dictionaries and their normalized descriptors. Strict loading
  remaps the published checkpoint's `permutation_network.*` and
  `overlap_diffusion_net.*` keys under `overlap_predictor.`.
- Keep precomputed 768-dimensional DINO descriptors as inputs to the
  trainable EchoMatch feature extractor. The EchoMatch weights, optimizer,
  and counters start fresh.
- Keep the original 50-mode FM solver and GT Y-mask pruning
  (`prune_strategy: f2_only`) during training. The correspondence objective
  retains the FM loss (weight 1) and within-shape and cross-shape InfoNCE
  (0.1 each). Evaluate direct descriptor nearest neighbors (`point`) and
  unmasked FM (`fmap`). Use the dataset geodesics and pool errors over all
  annotated correspondences. Matching AUC integrates the direct-matching
  PCK curve on the existing 40-point grid from 0 to 0.25.
- Use seed 1234, batch size 1, Adam at `1e-3` with betas `[0.9, 0.99]`, and
  StepLR with step size 100 and gamma 0.5. Train for **198 epochs** with
  the standard trainer. On the public split's 304 directional train maps,
  this gives 60,192 iterations. Evaluate the periodic checkpoint at
  **60,000 iterations**, the budget in sections A and H of the paper
  supplement. The supplement reports 273 training pairs, so the
  iteration count is the common reference.
  [EchoMatch paper and supplement](https://xieyizheng.com/media/papers/echomatch/echo_match_paper_reduced.pdf)
- Evaluate the 100 official test maps after training. The standard
  trainer runs its end-of-training validation at iteration 60,192.
  Use the separately evaluated `60000.pth` results for the comparison.
  Compare `point` geodesic error and PCK with the paper where the
  matching rule, normalization, and aggregation agree. The paper uses
  predicted overlap for its reported matching. Report `fmap` mean error
  separately.
  [EchoMatch, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_EchoMatch_Partial-to-Partial_Shape_Matching_via_Correspondence_Reflection_CVPR_2025_paper.html)

## Verification

On synthetic pairs with PyTorch 2.3.0 CPU, the default predictor matched
master `c4236ff` exactly for forward results, losses, gradients, and one
optimizer update. All six bundled EchoMatch checkpoints loaded strictly.
The matching-only train and test pipelines were exercised on cached
synthetic descriptors, including checkpoint reload and PCK export. The
official PARTIALSMAL run on Marvin is pending.

## Commands

Run the synthetic CPU smoke test from the repository root in the existing
Python environment. It needs no PARTIALSMAL data or GPU allocation.

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -p test_echomatch_overlap_predictor.py -v
```

On Marvin, run from the repository root with the official splits available
under `../data/PARTIALSMAL`. The launcher loads the Python module, activates
the existing venv, and prints the Git revision, map-set digests, and cache
counts. It reuses the DINO and geometry caches in each split where
available. Regenerating missing DINO features requires a compatible
PyTorch3D renderer and a populated or reachable Torch Hub cache. DINOv2's
Hub revision is unpinned, so record the cache version for reproducibility.
The launcher reserves one A40 for up to eight hours in `mlgpu_short`.
The previous warm-cache DINO run suggests roughly 2 h 45 min for 60,000
iterations, with actual runtime depending on cache hits and GPU throughput.
The evaluation job starts after training completes successfully.

```bash
scratch=/lustre/scratch/data/s23ahigu_hpc-echomatch
export TORCH_HOME="$scratch/torch-cache"
train_job=$(sbatch --parsable --account=ag_ifi_blum \
  --output="$scratch/echo-dino-matching-only-train-%j.out" \
  train_partialsmal_dino_matching_only.sbatch train) &&
eval_job=$(sbatch --parsable --account=ag_ifi_blum \
  --partition=mlgpu_short --time=00:20:00 --dependency="afterok:${train_job%%;*}" \
  --output="$scratch/echo-dino-matching-only-test-%j.out" \
  train_partialsmal_dino_matching_only.sbatch evaluate) &&
printf 'Training: %s\nEvaluation: %s\n' "$train_job" "$eval_job"
```

The run directory is
`experiments/echo_match/echo_match_psmal_dino_matching_only/`, and the
default evaluation output directory is
`results/echo_match/echo_match_psmal_dino_matching_only50_/`.
The launcher evaluates `models/60000.pth` and logs its SHA-256.
`models/final.pth` contains the epoch-198 weights at iteration 60,192.
An optional path after `evaluate` selects an explicit checkpoint and
uses a distinct result directory.

To compare unmasked `fmap` and `fmap_oracle` on the same test maps in
one job, use `evaluate-oracle` in place of `evaluate`. This applies the
GT overlap mask only to Y's descriptors before the oracle solver call.
Both variants use `60000.pth` by default. Compare their logged mean
geodesic errors. The logged `Val auc` refers to direct `point` matching.
The oracle mode saves results under a separate configuration suffix.
