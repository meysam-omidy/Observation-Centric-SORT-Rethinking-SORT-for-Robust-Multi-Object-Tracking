"""Contracts for the paper-update tracker options. Run: python -m pytest -q"""
import argparse
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from kalman_filter import learned_process_noise_matrix
from ocsort import OCSORTTracker
from track import Track
from track_state import StateDeleted, StateLost
from utils import (
    compute_adaptive_kalman_features, compute_detection_context, get_dict_item, get_dict_key,
)
import run_tracker

MOTION_PREDICTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'motion-predictor')


def reference_features(bboxes, scores, observed, max_gap_norm=30.0):
    """The original loop implementation, kept to pin the vectorized builder."""
    bboxes = np.asarray(bboxes, dtype=float).reshape(-1, 4)
    n = len(bboxes)
    feat = np.zeros((n, 15), dtype=np.float32)
    for i in range(n):
        if not observed[i]:
            continue
        feat[i, :4] = bboxes[i]
        feat[i, 12] = scores[i]
        if i >= 1 and observed[i - 1]:
            feat[i, 4:8] = bboxes[i] - bboxes[i - 1]
            if i >= 2 and observed[i - 2]:
                feat[i, 8:12] = (bboxes[i] - bboxes[i - 1]) - (bboxes[i - 1] - bboxes[i - 2])
    gap = 0.0
    for i in range(n):
        gap = 0.0 if observed[i] else gap + 1.0
        feat[i, 13] = min(gap / max_gap_norm, 1.0)
        feat[i, 14] = 1.0 if observed[i] else 0.0
    return feat


class FeatureAndHelperTest(unittest.TestCase):
    def test_vectorized_features_match_the_original_loop(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            n = int(rng.integers(1, 40))
            boxes, scores = rng.random((n, 4)), rng.random(n)
            observed = rng.random(n) < rng.uniform(0.2, 1.0)
            np.testing.assert_array_equal(
                compute_adaptive_kalman_features(boxes, scores, observed),
                reference_features(boxes, scores, observed),
            )

    @unittest.skipUnless(os.path.isdir(MOTION_PREDICTOR), 'motion-predictor checkout not found')
    def test_training_and_tracker_builders_are_identical(self):
        sys.path.insert(0, MOTION_PREDICTOR)
        try:
            from adaptive_kalman_dataset import (
                build_adaptive_kalman_features, compute_detection_context as train_context,
            )
        finally:
            sys.path.remove(MOTION_PREDICTOR)
        rng = np.random.default_rng(5)
        for _ in range(50):
            n = int(rng.integers(2, 40))
            boxes = rng.random((n, 4)) * 500
            boxes[:, 2:] += boxes[:, :2]
            np.testing.assert_array_equal(
                compute_detection_context(boxes, 1000, 700), train_context(boxes, 1000, 700)
            )
            context = rng.random((n, 3))
            observed = rng.random(n) < 0.7
            np.testing.assert_array_equal(
                compute_adaptive_kalman_features(boxes, np.ones(n), observed, context=context),
                build_adaptive_kalman_features(boxes, np.ones(n), observed, context=context),
            )

    def test_dict_lookup_from_either_end(self):
        data = {k: k * 10 for k in range(7)}
        for index in (-1, -2, -7, 0, 3, 6):
            self.assertEqual(get_dict_item(data, index), list(data.values())[index])
            self.assertEqual(get_dict_key(data, index), list(data.keys())[index])
        with self.assertRaises(IndexError):
            get_dict_item(data, -8)

    def test_seven_dim_q_scales_velocity_variances_to_pixels(self):
        var = np.array([1e-6, 2e-6, 3e-6, 4e-6, 1e-5, 2e-5, 3e-5])
        Q = learned_process_noise_matrix(var, np.array([100, 100, 40, 80]), 1000, 500)
        np.testing.assert_allclose(np.diag(Q)[4:], [1e-5 * 1000 ** 2, 2e-5 * 500 ** 2,
                                                    3e-5 * (1000 * 500) ** 2])
        Q4 = learned_process_noise_matrix(var[:4], np.array([100, 100, 40, 80]), 1000, 500)
        np.testing.assert_allclose(np.diag(Q4)[4:], 0.01)
        np.testing.assert_allclose(np.diag(Q)[:4], np.diag(Q4)[:4])


class TrackOptionTest(unittest.TestCase):
    def _track(self, **config):
        return Track(np.array([100.0, 100.0, 140.0, 200.0]), 0.9, 1, 1,
                     {'image_width': 1000, 'image_height': 500, **config})

    def _coast(self, track, frames):
        for _ in range(frames):
            track.predict()
            track.set_prediction_from_motion(np.asarray(track.bbox), track.score)

    def test_uncertainty_lifecycle_replaces_max_age(self):
        fixed = self._track(max_age=30)
        adaptive = self._track(max_age=30, max_position_uncertainty=0.05, uncertainty_max_age=90)
        for track in (fixed, adaptive):
            track.update(np.array([102.0, 100.0, 142.0, 200.0]), 0.9)
        self._coast(fixed, 5)
        self._coast(adaptive, 5)
        self.assertTrue(fixed.is_valid)
        self.assertEqual(adaptive.state, StateLost)
        self.assertGreater(adaptive.position_uncertainty, 0.05)
        self.assertFalse(adaptive.is_valid)

    def test_recorded_innovations_give_a_finite_nis(self):
        track = self._track(record_innovations=True)
        self._coast(track, 1)
        track.update(np.array([104.0, 101.0, 144.0, 201.0]), 0.8)
        record = track.innovation_records[-1]
        self.assertTrue(np.isfinite(record['nis']) and record['nis'] >= 0)
        self.assertEqual(record['gap'], 1)
        self.assertEqual(len(record['innovation_variance']), 4)

    def test_disabling_learned_r_keeps_the_fixed_r(self):
        track = self._track(use_learned_r=False)
        self.assertIsNone(track.measurement_noise(np.array([1, 1, 5, 5.0]), 0.9, np.ones(4) * 1e-4))
        learned = self._track()
        self.assertIsNotNone(learned.measurement_noise(np.array([1, 1, 5, 5.0]), 0.9, np.ones(4) * 1e-4))


class AssociationOptionTest(unittest.TestCase):
    def test_box_expansion_uses_predicted_position_std(self):
        tracker = OCSORTTracker({'association_box_expansion_sigma': 2.0})
        track = SimpleNamespace(kf=SimpleNamespace(P=np.diag([9.0, 4.0, 1, 1, 1, 1, 1])))
        boxes = np.array([[10.0, 10.0, 20.0, 30.0]])
        np.testing.assert_allclose(tracker._expanded_track_boxes([track], boxes),
                                   [[4.0, 6.0, 26.0, 34.0]])
        tracker.config.association_box_expansion_sigma = 0.0
        self.assertIs(tracker._expanded_track_boxes([track], boxes), boxes)

    def test_row_relative_logdet_ignores_invalid_cells(self):
        values = np.array([[3.0, 1.0, 9.0], [np.nan, 5.0, 2.0]])
        valid = np.array([[True, True, False], [True, True, True]])
        np.testing.assert_allclose(OCSORTTracker._row_relative(values, valid),
                                   [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])


class ConfigPresetTest(unittest.TestCase):
    def _parser(self):
        p = argparse.ArgumentParser()
        p.add_argument('--config')
        p.add_argument('--max_age', type=int, default=30)
        p.add_argument('--use_byte', action='store_true', default=False)
        return p

    def test_preset_sets_defaults_and_flags_override(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump({'_comment': 'ignored', 'max_age': 50, 'use_byte': True}, f)
        try:
            p = self._parser()
            run_tracker.load_config_defaults(p, ['--config', f.name])
            args = p.parse_args(['--config', f.name, '--max_age', '10'])
            self.assertEqual(args.max_age, 10)
            self.assertTrue(args.use_byte)
        finally:
            os.remove(f.name)

    def test_unknown_preset_key_is_rejected(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump({'max_agee': 50}, f)
        try:
            with self.assertRaises(SystemExit):
                run_tracker.load_config_defaults(self._parser(), ['--config', f.name])
        finally:
            os.remove(f.name)

    def test_shipped_presets_only_use_known_arguments(self):
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
        if not os.path.isdir(config_dir):
            self.skipTest('no configs directory')
        source = open(run_tracker.__file__, encoding='utf-8').read()
        for name in os.listdir(config_dir):
            if not name.endswith('.json'):
                continue
            preset = json.load(open(os.path.join(config_dir, name), encoding='utf-8'))
            for key in preset:
                if not key.startswith('_'):
                    self.assertIn(f"'{key}'", source.replace('--', "'").replace("dest='", "'"),
                                  f'{name}: {key}')


if __name__ == '__main__':
    unittest.main()
