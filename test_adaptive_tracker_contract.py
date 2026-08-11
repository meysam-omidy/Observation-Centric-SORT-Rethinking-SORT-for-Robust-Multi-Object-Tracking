import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

import run_tracker
import run_qr_scale_sweep
from ocsort import OCSORTTracker, OCSORTTrackerConfig
from run_tracker import LockedMotionPredictorEngine
from track import Track, TrackConfig, TrackHistory, TrackHistoryItem
from track_state import StateTracking
from utils import BBOX


class AdaptiveTrackerContractTest(unittest.TestCase):
    def _tracker(self, history_len=3):
        tracker = object.__new__(OCSORTTracker)
        tracker.config = SimpleNamespace(image_width=100, image_height=100)
        tracker.motion_engine = SimpleNamespace(
            history_len=history_len, max_gap_norm=30.0
        )
        return tracker

    @staticmethod
    def _item(x, observed):
        return TrackHistoryItem(BBOX([x, x, 10, 10]), 0.9, observed=observed)

    def test_context_is_fixed_length_and_preserves_lost_frames(self):
        tracker = self._tracker(history_len=3)
        items = [
            self._item(10, True),
            self._item(11, True),
            self._item(12, True),
            self._item(13, False),
            self._item(14, False),
        ]
        track = SimpleNamespace(k_last_updates=items)
        _, _, observed, features = tracker._adaptive_context(track)
        self.assertEqual(features.shape, (3, 15))
        np.testing.assert_array_equal(observed, [True, False, False])
        np.testing.assert_allclose(features[:, 13], [0.0, 1 / 30.0, 2 / 30.0])
        np.testing.assert_array_equal(features[:, 14], [1.0, 0.0, 0.0])

    def test_current_measurement_resets_gap_for_r(self):
        tracker = self._tracker(history_len=3)
        items = [self._item(10, True), self._item(11, False), self._item(12, False)]
        context = tracker._adaptive_context(SimpleNamespace(k_last_updates=items))
        measurement = tracker._measurement_feature(
            context, np.array([15, 15, 25, 25], dtype=float), 0.7
        )
        self.assertEqual(measurement[13], 0.0)
        self.assertEqual(measurement[14], 1.0)
        self.assertAlmostEqual(float(measurement[12]), 0.7, places=6)

    def test_noise_scale_defaults_and_validation(self):
        self.assertEqual(OCSORTTrackerConfig().q_scale, 1.0)
        self.assertEqual(OCSORTTrackerConfig().r_scale, 1.0)
        self.assertEqual(TrackConfig().q_scale, 1.0)
        self.assertEqual(TrackConfig().r_scale, 1.0)
        with self.assertRaises(ValidationError):
            OCSORTTrackerConfig(q_scale=0.0)
        with self.assertRaises(ValidationError):
            TrackConfig(r_scale=-1.0)

    def test_tracker_propagates_scales_to_new_tracks(self):
        tracker = OCSORTTracker({
            'q_scale': 0.25,
            'r_scale': 0.05,
            'motion': {'enabled': False},
        })
        tracker.update(np.array([[10.0, 10.0, 30.0, 30.0, 0.9]]))
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.tracks[0].config.q_scale, 0.25)
        self.assertEqual(tracker.tracks[0].config.r_scale, 0.05)

    def test_q_scale_multiplies_the_complete_process_matrix(self):
        class FakeKalman:
            def __init__(self):
                self.x = np.zeros((7, 1), dtype=float)
                self.received_q = None

            def predict(self, Q=None):
                self.received_q = Q

        track = object.__new__(Track)
        track.config = SimpleNamespace(
            image_width=100,
            image_height=100,
            use_kalman=True,
            kalman_fusion_blend=0.0,
            q_scale=0.25,
        )
        track.kf = FakeKalman()
        track.history = TrackHistory()
        track.entered_frame = 1
        track.frame_count = 0

        unscaled_q = np.diag(np.arange(1.0, 8.0))
        with patch('track.learned_process_noise_matrix', return_value=unscaled_q):
            track.set_prediction_from_motion(
                np.array([20.0, 20.0, 10.0, 10.0]),
                score=0.9,
                var_q=np.ones(4),
            )

        np.testing.assert_allclose(track.kf.received_q, 0.25 * unscaled_q)

    def test_r_scale_multiplies_the_complete_measurement_matrix(self):
        class FakeKalman:
            def __init__(self):
                self.received_r = None

            def update(self, _z, R=None):
                self.received_r = R

        track = object.__new__(Track)
        track.config = SimpleNamespace(
            image_width=100,
            image_height=100,
            use_confidence_r=False,
            r_scale=0.05,
            reupdate_type=None,
        )
        track.kf = FakeKalman()
        track.history = TrackHistory()
        track.entered_frame = 1
        track.frame_count = 0
        track.age = 0
        track.state = StateTracking
        track.logs = {'max_time_lost': 0}

        unscaled_r = np.diag(np.arange(1.0, 5.0))
        with patch('track.learned_measurement_noise_matrix', return_value=unscaled_r):
            track.update(
                np.array([10.0, 10.0, 30.0, 30.0]),
                score=0.9,
                var_r=np.ones(4),
            )

        np.testing.assert_allclose(track.kf.received_r, 0.05 * unscaled_r)


class ParallelSequenceContractTest(unittest.TestCase):
    def test_scale_sweep_forwards_each_selected_sequence(self):
        args = SimpleNamespace(
            dataset='DanceTrack',
            split='val',
            seqs=['dancetrack0001', 'dancetrack0002'],
            datasets_dir='datasets',
            detections_dir='detections',
            weights_path='weights.pth',
            sequence_workers=2,
            device=None,
            tracker_args=[],
        )
        experiment = run_qr_scale_sweep.learned_experiment('test', 1.0, 0.1)
        with patch('run_qr_scale_sweep.subprocess.run') as subprocess_run:
            run_qr_scale_sweep.run_experiment(experiment, args)

        command = subprocess_run.call_args.args[0]
        seqs_index = command.index('--seqs')
        self.assertEqual(
            command[seqs_index + 1:seqs_index + 3],
            args.seqs,
        )

    def test_shared_motion_engine_serializes_inference(self):
        class FakeEngine:
            marker = 'forwarded'

            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.state_lock = threading.Lock()

            def predict_q_batch(self, value):
                with self.state_lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.state_lock:
                    self.active -= 1
                return value

        engine = FakeEngine()
        locked = LockedMotionPredictorEngine(engine)
        threads = [
            threading.Thread(target=locked.predict_q_batch, args=(i,))
            for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(engine.max_active, 1)
        self.assertEqual(locked.marker, 'forwarded')

    def test_main_runs_unique_sequences_concurrently(self):
        args = SimpleNamespace(
            seqs=['seq-a', 'seq-b', 'seq-a', 'seq-c'],
            datasets_dir='unused',
            dataset='DanceTrack',
            split='val',
            sequence_workers=3,
            tracker_name='parallel-test',
            motion_enabled=False,
            evaluate=False,
        )
        active = 0
        max_active = 0
        seen = []
        state_lock = threading.Lock()

        def fake_run(sequence, _args, _engine):
            nonlocal active, max_active
            with state_lock:
                seen.append(sequence)
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1

        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                with patch('run_tracker.run', side_effect=fake_run):
                    run_tracker.main(args)
                seqmap = os.path.join(
                    temporary_directory, 'trackeval', 'seqmap',
                    'dancetrack', 'custom.txt',
                )
                with open(seqmap, encoding='utf-8') as seqmap_file:
                    self.assertEqual(
                        seqmap_file.read().splitlines(),
                        ['name', 'seq-a', 'seq-b', 'seq-c'],
                    )
            finally:
                os.chdir(previous_cwd)

        self.assertCountEqual(seen, ['seq-a', 'seq-b', 'seq-c'])
        self.assertGreater(max_active, 1)


if __name__ == '__main__':
    unittest.main()
