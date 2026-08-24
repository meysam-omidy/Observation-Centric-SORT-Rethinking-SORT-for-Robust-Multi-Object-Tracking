import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

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


if __name__ == "__main__":
    unittest.main()
