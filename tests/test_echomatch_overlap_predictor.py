"""Regression checks for EchoMatch's optional overlap predictor on CPU."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import matplotlib
import torch
import yaml

matplotlib.use("Agg")

from models import build_model
from utils.logger import AvgTimer


ROOT = Path(__file__).resolve().parents[1]
NETWORK = "echo_match_net"


def make_options(is_train=True, matching_only=False):
    split = "train" if is_train else "test"
    suffix = "_matching_only" if matching_only else ""
    config_path = ROOT / "options" / "echo_match" / split / f"echo_match_psmal_dino{suffix}.yaml"
    options = yaml.safe_load(config_path.read_text())
    options.update(num_gpu=0, is_train=is_train, dist=False, rank=0, world_size=1)
    return options


def make_batch():
    generator = torch.Generator().manual_seed(437)
    batch = {}
    for shape, count, corr in (("first", 6, [0, 2, 4]), ("second", 8, [1, 3, 5])):
        verts = torch.randn(count, 3, generator=generator)
        eigenvectors = torch.linalg.qr(torch.randn(count, 4, generator=generator)).Q
        eigenvalues = torch.linspace(0.1, 1.0, eigenvectors.shape[1])
        mask = torch.zeros(count)
        mask[corr] = 1
        batch[shape] = {
            "name": [shape],
            "verts": verts.unsqueeze(0),
            "dino": 0.5 * torch.randn(1, count, 768, generator=generator),
            "corr": torch.tensor([corr]),
            "dist": torch.cdist(verts, verts).unsqueeze(0),
            "partiality_mask": mask.unsqueeze(0),
            "evals": eigenvalues.unsqueeze(0),
            "evecs": eigenvectors.unsqueeze(0),
            "evecs_trans": eigenvectors.T.unsqueeze(0),
            "operators": {
                "mass": torch.ones(1, count),
                "L": torch.eye(count).unsqueeze(0),
                "evals": eigenvalues.unsqueeze(0),
                "evecs": eigenvectors.unsqueeze(0),
                "gradX": 0.04 * torch.randn(1, count, count, generator=generator),
                "gradY": 0.04 * torch.randn(1, count, count, generator=generator),
            },
        }
    return batch


class OptionalOverlapPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(1234)

    def test_published_dino_weights_load_strictly_with_predictor_default(self):
        options = make_options(is_train=False)
        checkpoint_path = ROOT / options["path"].pop("resume_state")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_model(options)
        net = model.networks[NETWORK]
        self.assertIsNotNone(net.overlap_predictor)
        published = state["networks"][NETWORK]
        net.load_state_dict(published, strict=True)

        loaded = net.state_dict()
        for name, expected in published.items():
            name = name.removeprefix("module.")
            if name.startswith(("permutation_network.", "overlap_diffusion_net.")):
                name = "overlap_predictor." + name
            torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0)
        model.eval()
        with torch.no_grad():
            result = model.validate_single(make_batch(), AvgTimer())
        self.assertEqual(set(result["p2p_variants"]), {"point", "fmap"})
        for key, count in (("overlap_score12", 6), ("overlap_score21", 8)):
            self.assertEqual(result[key].shape, (count,))
            self.assertTrue(torch.isfinite(result[key]).all())

    def test_matching_only_trains_with_gt_fm_mask_and_without_bce(self):
        options = make_options(matching_only=True)
        self.assertIs(options["path"]["resume"], False)
        self.assertNotIn("resume_state", options["path"])
        self.assertEqual(options["train"]["total_iters"], 60000)
        self.assertNotIn("test_dataset", options["datasets"])
        self.assertIsNone(options["val"])
        model = build_model(options)
        net = model.networks[NETWORK]
        self.assertIsNone(net.overlap_predictor)
        self.assertFalse(any(name.startswith("overlap_predictor.") for name in net.state_dict()))
        self.assertEqual((model.curr_epoch, model.curr_iter), (0, 0))
        optimizer = model.optimizers[NETWORK]
        optimizer_ids = [id(param) for group in optimizer.param_groups for param in group["params"]]
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer_ids, [id(param) for param in net.feature_extractor.parameters()])
        batch = make_batch()

        features, solver_inputs = [], []
        feature_hook = net.feature_extractor.register_forward_hook(
            lambda module, args, output: features.append(output.detach())
        )
        solver_hook = net.fmreg_net.register_forward_pre_hook(
            lambda module, args: solver_inputs.append(args[1].detach())
        )
        try:
            with patch.object(model.losses["echo_match_loss"].binary_loss, "forward",
                              side_effect=AssertionError("Overlap BCE must be skipped")):
                model.feed_data(batch)
        finally:
            feature_hook.remove()
            solver_hook.remove()

        expected_y = features[1] * batch["second"]["partiality_mask"].unsqueeze(-1)
        self.assertEqual(len(solver_inputs), 1)
        torch.testing.assert_close(solver_inputs[0], expected_y, rtol=0, atol=0)
        self.assertEqual(set(model.loss_metrics), {"fmap_loss", "nce_loss"})
        for loss in model.loss_metrics.values():
            self.assertTrue(torch.isfinite(loss))
        parameter = net.feature_extractor.first_lin.weight
        initial_weight = parameter.detach().clone()
        model.optimize_parameters()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertFalse(torch.equal(parameter.detach(), initial_weight))

    def test_matching_only_eval_without_masks_keeps_matching_outputs(self):
        options = make_options(is_train=False, matching_only=True)
        self.assertTrue(options["path"]["require_checkpoint"])
        options["path"].pop("resume_state")
        model = build_model(options)
        model.eval()
        batch = make_batch()
        missing_masks = deepcopy(batch)
        for shape in missing_masks.values():
            shape.pop("partiality_mask")
        with torch.no_grad():
            reference = model.validate_single(batch, AvgTimer())
            result = model.validate_single(missing_masks, AvgTimer())

        self.assertNotIn("overlap_score12", result)
        self.assertNotIn("overlap_score21", result)
        self.assertEqual(result["p2p_variants"].keys(), {"fmap", "point"})
        torch.testing.assert_close(result["p2p"], reference["p2p"], rtol=0, atol=0)
        torch.testing.assert_close(
            result["p2p_variants"], reference["p2p_variants"], rtol=0, atol=0
        )

        with TemporaryDirectory() as directory, patch("models.partial_base_model.wandb.log") as log:
            model.opt["path"]["results_root"] = directory
            (Path(directory) / "pcks").mkdir()
            model.validation([missing_masks], tb_logger=None, update=False)
            saved_files = {path.name for path in Path(directory).iterdir()}
            self.assertTrue({"geo_errors.pkl", "p2p.pkl", "pck.png"}.issubset(saved_files))
            self.assertTrue((Path(directory) / "pcks" / "first-second.jpg").is_file())
            self.assertTrue({"overlap_score12.pkl", "overlap_score21.pkl", "pck_all.png",
                             "iou_curve.png", "tikz_plot.txt"}.isdisjoint(saved_files))
            summaries = log.call_args.args[0]
            self.assertIn("test avg error", summaries)
            self.assertIn("test auc", summaries)
            self.assertNotIn("test miou1", summaries)
            self.assertNotIn("test b_acc1", summaries)
            self.assertNotIn("test auc_all", summaries)

    def test_matching_only_periodic_validation_selects_on_point_error(self):
        options = make_options(matching_only=True)
        options["val"] = make_options()["val"]
        model = build_model(options)
        model.curr_iter = options["val"]["val_freq"]
        tb_logger = Mock()
        with patch.object(model, "save_model") as save_model:
            model.validation([make_batch()], tb_logger=tb_logger)
        values = {record.args[0]: record.args[1] for record in tb_logger.add_scalar.call_args_list}
        self.assertEqual(model.best_metric, values["val point avg error"])
        self.assertEqual(model.best_metric, values["val avg error"])
        self.assertIn("val fmap avg error", values)
        self.assertNotIn("val miou1", values)
        save_model.assert_called_once_with(net_only=False, best=True, save_filename="best.pth")


if __name__ == "__main__":
    unittest.main()
