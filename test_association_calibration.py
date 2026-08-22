import unittest

import numpy as np

from evaluate_association_calibration import (
    CHI2_2D,
    RANKING_METHODS,
    _one_to_one_matches,
    _rank_costs,
    _ranking_summary,
    _summary,
)


class AssociationCalibrationTest(unittest.TestCase):
    def test_one_to_one_matching_does_not_reuse_a_detection(self):
        gt = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [8.0, 0.0, 18.0, 10.0],
        ])
        detections = np.array([
            [1.0, 0.0, 11.0, 10.0],
            [8.0, 0.0, 18.0, 10.0],
        ])
        matches = _one_to_one_matches(gt, detections, min_iou=0.5)
        self.assertEqual(matches, [(0, 0), (1, 1)])

    def test_summary_matches_chi_square_reference_for_reference_points(self):
        summary = _summary([CHI2_2D["q50"], CHI2_2D["q95"]])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["coverage_50"], 0.5)
        self.assertEqual(summary["coverage_95"], 1.0)

    def test_summary_applies_global_covariance_scale(self):
        raw = _summary([2.0, 4.0])
        scaled = _summary([2.0, 4.0], scale=2.0)
        self.assertAlmostEqual(raw["mean_d2"], 3.0)
        self.assertAlmostEqual(scaled["mean_d2"], 1.5)

    def test_ranking_gives_correct_candidate_rank_and_pairwise_credit(self):
        ranking = _rank_costs(np.array([0.4, 0.1, 0.7]), correct_index=0)
        self.assertEqual(ranking["rank"], 2)
        self.assertFalse(ranking["top_1"])
        self.assertAlmostEqual(ranking["pairwise_preference"], 0.5)
        self.assertAlmostEqual(ranking["margin_to_nearest_competitor"], -0.3)

    def test_ranking_summary_handles_single_and_ambiguous_candidates(self):
        def methods(rank: int, candidates: int) -> dict:
            return {
                name: {
                    "rank": rank,
                    "top_1": rank == 1,
                    "strict_top_1": rank == 1,
                    "reciprocal_rank": 1.0 / rank,
                    "pairwise_preference": 1.0 if rank == 1 else 0.0,
                    "margin_to_nearest_competitor": None if candidates == 1 else 0.2,
                    "top_candidate_local_index": 0,
                }
                for name in RANKING_METHODS
            }

        summary = _ranking_summary([
            {"candidate_count": 1, "methods": methods(1, 1)},
            {"candidate_count": 2, "methods": methods(2, 2)},
        ])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["methods"]["iou_only"]["mean_rank"], 1.5)
        self.assertEqual(summary["methods"]["iou_only"]["top_1_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
