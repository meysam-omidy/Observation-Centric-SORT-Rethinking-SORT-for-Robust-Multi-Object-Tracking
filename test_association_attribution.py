import unittest
from types import SimpleNamespace

import numpy as np

from analyze_tracking import AssociationAttribution
from ocsort import OCSORTTracker


def _event(track_ids, detection_indices, costs, matches):
    costs = np.asarray(costs, dtype=float)
    return {
        "frame": 2,
        "phase": 1,
        "track_ids": track_ids,
        "track_states": ["Tracking"] * len(track_ids),
        "detection_indices": np.asarray(detection_indices, dtype=int),
        "scores": np.ones(len(detection_indices)),
        "iou": 1.0 - costs,
        "direction_cost": np.zeros_like(costs),
        "cost": costs,
        "valid_pairs": np.ones_like(costs, dtype=bool),
        "matches": matches,
    }


class AssociationAttributionTest(unittest.TestCase):
    def setUp(self):
        self.gt = [(7, np.array([0.0, 0.0, 10.0, 10.0]))]
        self.detections = np.array([
            [0.0, 0.0, 10.0, 10.0, 0.9],  # oracle detection for GT 7
            [20.0, 0.0, 30.0, 10.0, 0.9],
        ])
        self.tracker = SimpleNamespace(tracks=[], get_outputs=lambda: [])

    def test_labels_lower_cost_wrong_candidate_as_local_ranking_failure(self):
        attribution = AssociationAttribution(match_iou=0.5)
        attribution.track_to_gt = {1: 7}
        attribution.consume_frame(
            2, self.gt, self.detections,
            [_event([1], [0, 1], [[0.4, 0.1]], [[0, 1]])],
            self.tracker,
        )
        failure = attribution.failures[0]
        self.assertEqual(failure["label"], "LOCAL_RANKING_FAILURE")
        self.assertEqual(failure["expected_detection_index"], 1)
        self.assertEqual(failure["assigned_detection_index"], 2)
        self.assertEqual(failure["local_rank"], 2)

    def test_labels_locally_best_expected_detection_lost_by_assignment_as_global(self):
        attribution = AssociationAttribution(match_iou=0.5)
        attribution.track_to_gt = {1: 7, 2: 8}
        gt = [
            *self.gt,
            (8, np.array([20.0, 0.0, 30.0, 10.0])),
        ]
        attribution.consume_frame(
            2, gt, self.detections,
            [_event(
                [1, 2], [0, 1],
                [[0.1, 0.2], [0.05, 0.8]],
                [[0, 1], [1, 0]],
            )],
            self.tracker,
        )
        failure = attribution.failures[0]
        self.assertEqual(failure["label"], "GLOBAL_DISPLACEMENT")
        self.assertEqual(failure["local_rank"], 1)
        self.assertEqual(failure["expected_detection_owner_track_id"], 2)
        self.assertEqual(failure["conflict"]["type"], "TWO_WAY_SWAP")
        self.assertEqual(
            failure["conflict"]["costs"]["a_to_expected_a"]["cost"], 0.1
        )


class AssociationDiagnosticsContractTest(unittest.TestCase):
    def test_tracker_exposes_phase_diagnostics_only_when_requested(self):
        tracker = OCSORTTracker({
            "motion": {"enabled": False},
            "collect_association_diagnostics": True,
        })
        detections = np.array([
            [10.0, 10.0, 30.0, 30.0, 0.9],
            [50.0, 10.0, 70.0, 30.0, 0.2],
        ])
        tracker.update(detections)
        phase_one = next(event for event in tracker.last_association_diagnostics if event["phase"] == 1)
        np.testing.assert_array_equal(phase_one["detection_indices"], [0])
        self.assertEqual(phase_one["cost"].shape, (0, 1))
        self.assertEqual(phase_one["valid_pairs"].dtype, np.dtype(bool))

    def test_oracle_callback_can_force_only_an_already_valid_pair(self):
        tracker = OCSORTTracker({"motion": {"enabled": False}})
        tracker.update(np.array([[10.0, 10.0, 30.0, 30.0, 0.9]]))
        track = tracker.tracks[0]

        calls = []

        def force_second_candidate(**event):
            calls.append(event)
            self.assertEqual(event["phase"], 1)
            self.assertTrue(event["valid_pairs"][0, 1])
            return [(0, 1)]

        matches, _, _ = tracker.associate(
            [track],
            np.array([
                [10.0, 10.0, 30.0, 30.0],
                [15.0, 10.0, 35.0, 30.0],
            ]),
            np.array([0.9, 0.9]),
            iou_threshold=0.2,
            phase=1,
            association_override=force_second_candidate,
        )

        self.assertEqual(matches, [[0, 1]])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
