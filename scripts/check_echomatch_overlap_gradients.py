"""Check whether the FM loss reaches the overlap predictor on PARTIALSMAL."""

import argparse
from itertools import islice
from pathlib import Path

import torch

from datasets import build_dataloader, build_dataset
from models import build_model
from utils.misc import set_random_seed
from utils.options import parse


def check_batch(model, batch, overlap_parameters, seed):
    """Compare the two training paths on the same batch and random draws."""
    source = batch["second"]["name"][0]
    target = batch["first"]["name"][0]
    for strategy in ("f2_only", "predicted_y"):
        set_random_seed(seed)
        model.prune_strategy = strategy
        model.feed_data(batch)
        loss = model.loss_metrics["fmap_loss"]
        model.loss_metrics.clear()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{source} -> {target}: {strategy} FM loss is not finite")

        gradients = torch.autograd.grad(loss, overlap_parameters, allow_unused=True)
        connected = [gradient for gradient in gradients if gradient is not None]
        if strategy == "f2_only":
            if connected:
                raise RuntimeError(f"{source} -> {target}: GT-masked FM loss reached the overlap predictor")
            print(f"{source} -> {target} | {strategy} | FM loss {loss.item():.6g} | overlap gradient absent")
        else:
            if not connected:
                raise RuntimeError(f"{source} -> {target}: predicted-mask FM loss did not reach the overlap predictor")
            norm = torch.linalg.vector_norm(torch.cat([gradient.reshape(-1) for gradient in connected]))
            if not torch.isfinite(norm) or norm.item() == 0:
                raise RuntimeError(f"{source} -> {target}: predicted-mask overlap gradient is zero or non-finite")
            print(
                f"{source} -> {target} | {strategy} | FM loss {loss.item():.6g} | "
                f"overlap gradient norm {norm.item():.6g} ({len(connected)}/{len(gradients)} tensors)"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", required=True, help="Training config")
    parser.add_argument("--override", help="Runtime config overrides")
    parser.add_argument("--samples", type=int, default=2, help="Number of directional training samples")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    opt = parse(args.opt, str(Path.cwd()), is_train=True, override_path=args.override)
    path = opt["path"]
    if path.get("resume") is not False:
        parser.error("Set path.resume: false to load network weights with fresh training state")
    checkpoint = path.get("resume_state")
    if not checkpoint or not Path(checkpoint).is_file():
        parser.error("Set path.resume_state to an existing checkpoint")
    if opt["type"] != "Echo_Match_Model" or opt["backend"] != "dp" or opt["num_gpu"] != 1:
        parser.error("Use an Echo_Match_Model training config with one GPU and backend dp")
    if not torch.cuda.is_available():
        parser.error("CUDA is required to load the published checkpoint")
    if opt["datasets"]["batch_size"] != 1:
        parser.error("The directional EchoMatch training batch size must be one")

    seed = opt["manual_seed"]
    set_random_seed(seed)
    opt.update(dist=False, rank=0, world_size=1)
    dataset_options = opt["datasets"]
    dataset = build_dataset(dataset_options["train_dataset"])
    if args.samples > len(dataset):
        parser.error(f"--samples exceeds the size of the train split ({len(dataset)})")
    loader = build_dataloader(
        dataset, {**dataset_options, "num_worker": 0}, phase="train", num_gpu=1, seed=seed
    )

    model = build_model(opt)
    overlap_head = model.networks["echo_match_net"].overlap_diffusion_net
    overlap_parameters = tuple(parameter for parameter in overlap_head.parameters() if parameter.requires_grad)
    if not overlap_parameters:
        raise RuntimeError("The overlap predictor has no trainable parameters")

    for index, batch in enumerate(islice(loader, args.samples)):
        check_batch(model, batch, overlap_parameters, seed + index)
    print(f"FM gradient check passed for {args.samples} training samples.")


if __name__ == "__main__":
    main()
