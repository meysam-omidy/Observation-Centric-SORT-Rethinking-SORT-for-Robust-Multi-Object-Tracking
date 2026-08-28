import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from unittest.mock import patch

from association_model import (
    ASSOCIATION_MODEL_VERSION,
    PAIR_FEATURE_DIM,
    AssociationMLP,
    AssociationScorerEngine,
)
from ocsort import OCSORTTracker


class AssociationModelContractTest(unittest.TestCase):
    def _checkpoint(self, directory: str) -> str:
        model = AssociationMLP()
        path = Path(directory) / "association.pth"
        torch.save({
            "association_model_version": ASSOCIATION_MODEL_VERSION,
            "model_state_dict": model.state_dict(),
            "feature_dim": PAIR_FEATURE_DIM,
            "hidden_dim": 64,
            "feature_mean": np.zeros(PAIR_FEATURE_DIM, dtype=np.float32),
            "feature_std": np.ones(PAIR_FEATURE_DIM, dtype=np.float32),
        }, path)
        return str(path)

    def test_engine_scores_batched_features(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = AssociationScorerEngine(self._checkpoint(directory), "cpu")
            scores = engine.predict_logits(np.zeros((3, PAIR_FEATURE_DIM), dtype=np.float32))
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.isfinite(scores).all())

    def test_tracker_observer_sees_only_existing_valid_candidates(self):
        events = []
        tracker = OCSORTTracker({"motion": {"enabled": False}})
        tracker.update(np.array([[0.0, 0.0, 20.0, 40.0, 0.9]]))
        tracker.update(
            np.array([[1.0, 1.0, 21.0, 41.0, 0.9]]),
            association_observer=lambda **event: events.append(event),
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["valid_pairs"].shape, (1, 1))
        self.assertTrue(event["valid_pairs"][0, 0])
        self.assertEqual(event["iou"].shape, (1, 1))

    def test_learned_association_requires_checkpoint_path(self):
        with self.assertRaisesRegex(ValueError, "association_weights_path"):
            OCSORTTracker({"motion": {"enabled": False}, "use_learned_association": True})

    def test_legacy_iou_gate_keeps_raw_invalid_cost_for_assignment(self):
        tracker = OCSORTTracker({
            "motion": {"enabled": False},
            "legacy_post_assignment_iou_gate": True,
        })
        tracker.init_track(np.array([0.0, 0.0, 10.0, 10.0]), 0.9)
        captured = []

        def capture(cost):
            captured.append(cost.copy())
            return [], [0], [0]

        with patch("ocsort.assignment", side_effect=capture):
            tracker.associate(
                tracker.tracks,
                np.array([[100.0, 100.0, 110.0, 110.0]]),
                np.array([0.9]),
                iou_threshold=0.2,
                phase=1,
            )
        self.assertEqual(len(captured), 1)
        self.assertLess(captured[0][0, 0], 1e6)

    def test_default_iou_gate_masks_invalid_cost_before_assignment(self):
        tracker = OCSORTTracker({"motion": {"enabled": False}})
        tracker.init_track(np.array([0.0, 0.0, 10.0, 10.0]), 0.9)
        captured = []
        with patch("ocsort.assignment", side_effect=lambda cost: (captured.append(cost.copy()) or ([], [0], [0]))):
            tracker.associate(
                tracker.tracks,
                np.array([[100.0, 100.0, 110.0, 110.0]]),
                np.array([0.9]),
                iou_threshold=0.2,
                phase=1,
            )
        self.assertEqual(captured[0][0, 0], 1e6)

    def test_listwise_logits_become_row_relative_softmax_residuals(self):
        # The association model is trained with one softmax group per source
        # track. A common negative offset must not make every candidate receive
        # the same positive penalty at inference.
        residual = OCSORTTracker._learned_association_residual_from_logits(
            logits=np.array([-10.0, -12.0, 4.0]),
            candidate_rows=np.array([0, 0, 1]),
            candidate_cols=np.array([0, 2, 1]),
            shape=(2, 3),
            residual_clip=1.0,
        )
        probability = np.exp(2.0) / (1.0 + np.exp(2.0))
        np.testing.assert_allclose(
            residual,
            [[0.5 - probability, 0.0, probability - 0.5], [0.0, 0.0, 0.0]],
        )
        self.assertAlmostEqual(residual[0].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
