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
from track_state import StateDeleted, StateTracking
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
        self.assertIsNone(OCSORTTrackerConfig(log_path=None).log_path)
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

    def test_no_motion_engine_bypasses_history_feature_window(self):
        tracker = OCSORTTracker({
            'motion': {'enabled': False},
            'update_window_start': 2,
            'update_window_end': 2,
        })
        detections = np.array([[10.0, 10.0, 30.0, 30.0, 0.9]])
        tracker.update(detections)
        tracker.update(detections)

        # Before the fast path, the next update would enter the legacy-model
        # batching branch and call compute_motion_features despite no engine.
        with patch('ocsort.compute_motion_features', side_effect=AssertionError):
            tracker.update(detections)

        self.assertEqual(len(tracker.tracks), 1)

    def test_mahalanobis_gate_rejects_an_implausible_overlapping_detection(self):
        tracker = OCSORTTracker({
            'motion': {'enabled': False},
            'use_confidence_r': True,
            'use_mahalanobis_gate': True,
            'mahalanobis_gate_threshold': 5.0,
        })
        tracker.update(np.array([[10.0, 10.0, 30.0, 30.0, 0.9]]))
        track = tracker.tracks[0]

        # This box still passes the normal 0.2 IoU gate, but its 10.5-pixel
        # innovation is too large for the track's tight confidence covariance.
        matches, unmatched_tracks, unmatched_detections = tracker.associate(
            [track],
            np.array([
                [10.5, 10.0, 30.5, 30.0],  # covariance-valid candidate
                [20.5, 10.0, 40.5, 30.0],  # overlaps, but outside the gate
            ]),
            np.array([0.9, 0.9]),
            iou_threshold=0.2,
            phase=1,
        )
        self.assertEqual(matches, [[0, 0]])
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_detections, [1])

    def test_mahalanobis_cost_does_not_enable_the_hard_gate(self):
        tracker = OCSORTTracker({
            'motion': {'enabled': False},
            'use_confidence_r': True,
            'use_mahalanobis_cost': True,
            'mahalanobis_cost_coefficient': 0.05,
            'mahalanobis_gate_threshold': 5.0,
        })
        tracker.update(np.array([[10.0, 10.0, 30.0, 30.0, 0.9]]))
        track = tracker.tracks[0]

        # The candidate is outside the configured gate, but cost-only mode must
        # leave it eligible for its otherwise-valid IoU association.
        matches, unmatched_tracks, unmatched_detections = tracker.associate(
            [track],
            np.array([[20.5, 10.0, 40.5, 30.0]]),
            np.array([0.9]),
            iou_threshold=0.2,
            phase=1,
        )
        self.assertEqual(matches, [[0, 0]])
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_detections, [])

    def test_mahalanobis_soft_cost_is_normalized_and_clamped(self):
        tracker = OCSORTTracker({
            'motion': {'enabled': False},
            'association_speed_direction_coefficient': 0.0,
            'use_mahalanobis_cost': True,
            'mahalanobis_cost_coefficient': 1.0,
            'mahalanobis_cost_reference': 10.0,
        })
        tracker.init_track(np.array([0.0, 0.0, 20.0, 20.0]), 0.9)
        observed_costs = []
        detections = np.array([
            [0.0, 0.0, 20.0, 20.0],
            [0.0, 0.0, 20.0, 20.0],
        ])

        with patch.object(
            tracker,
            '_mahalanobis_distances',
            return_value=np.array([[2.5, 100.0]]),
        ):
            tracker.associate(
                tracker.tracks,
                detections,
                np.array([0.9, 0.9]),
                iou_threshold=0.2,
                phase=1,
                association_observer=lambda **event: observed_costs.append(event['base_cost']),
            )

        self.assertEqual(len(observed_costs), 1)
        np.testing.assert_allclose(observed_costs[0], [[0.25, 1.0]])

    def test_mahalanobis_cost_falls_back_when_distance_is_non_finite(self):
        tracker = OCSORTTracker({
            'motion': {'enabled': False},
            'use_mahalanobis_cost': True,
            'mahalanobis_cost_coefficient': 0.02,
        })
        tracker.update(np.array([[10.0, 10.0, 30.0, 30.0, 0.9]]))
        track = tracker.tracks[0]

        # An unavailable covariance score must not reject an otherwise-valid
        # IoU candidate when hard gating was not requested.
        with patch.object(
            tracker,
            '_mahalanobis_distances',
            return_value=np.array([[np.inf]]),
        ):
            matches, unmatched_tracks, unmatched_detections = tracker.associate(
                [track],
                np.array([[10.5, 10.0, 30.5, 30.0]]),
                np.array([0.9]),
                iou_threshold=0.2,
                phase=1,
            )
        self.assertEqual(matches, [[0, 0]])
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_detections, [])

    def test_duplicate_birth_suppression_only_flags_a_mature_overlapping_track(self):
        tracker = OCSORTTracker({
            'motion': {'enabled': False},
            'suppress_duplicate_track_births': True,
            'duplicate_track_iou_threshold': 0.85,
            'duplicate_track_min_observations': 1,
        })
        tracker.frame_number = 1
        tracker.init_track(np.array([10.0, 10.0, 30.0, 30.0]), 0.9)

        self.assertTrue(tracker._is_duplicate_track_birth(
            np.array([10.5, 10.0, 30.5, 30.0])
        ))
        self.assertFalse(tracker._is_duplicate_track_birth(
            np.array([45.0, 10.0, 65.0, 30.0])
        ))

    def test_duplicate_cleanup_defers_retirement_until_the_next_predict_step(self):
        tracker = OCSORTTracker({
            'motion': {'enabled': False},
            'cleanup_duplicate_tracks': True,
            'duplicate_track_iou_threshold': 0.85,
            'duplicate_track_min_observations': 1,
            'duplicate_track_overlap_frames': 1,
        })
        tracker.frame_number = 1
        tracker.init_track(np.array([10.0, 10.0, 30.0, 30.0]), 0.9)
        tracker.init_track(np.array([10.5, 10.0, 30.5, 30.0]), 0.8)

        tracker._queue_duplicate_track_cleanup()
        self.assertEqual({track.id for track in tracker.tracks if track.state == StateTracking}, {1, 2})
        self.assertEqual(tracker._pending_duplicate_track_deletions, {2})

        tracker.predict_tracks()
        self.assertEqual(tracker.tracks[0].state, StateTracking)
        self.assertEqual(tracker.tracks[1].state, StateDeleted)
        self.assertEqual(tracker.tracks[1].exited_frame, 1)

    def test_mature_priority_cascades_before_younger_or_lost_tracks(self):
        tracker = OCSORTTracker({
            'motion': {'enabled': False},
            'prioritize_mature_tracks': True,
            'mature_track_min_observations': 3,
        })

        def fake_track(track_id, observation_count):
            return SimpleNamespace(
                id=track_id,
                state=StateTracking,
                observation_count=observation_count,
            )

        mature = fake_track(10, 3)
        younger = fake_track(20, 1)
        detections = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 0.0, 30.0, 10.0],
        ])
        scores = np.array([0.9, 0.9])
        with patch.object(
            tracker,
            'associate',
            side_effect=[
                ([[0, 0]], [], [1]),  # mature track claims detection 0
                ([[0, 0]], [], []),   # younger track receives remaining detection 1
            ],
        ) as associate:
            matches, unmatched_tracks, unmatched_detections = tracker._associate_confirmed_tracks(
                [mature, younger], detections, scores, 0.2, phase=1,
                detection_indices=np.array([4, 7]),
            )

        self.assertEqual(matches, [[0, 0], [1, 1]])
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_detections, [])
        self.assertEqual([track.id for track in associate.call_args_list[0].args[0]], [10])
        self.assertEqual([track.id for track in associate.call_args_list[1].args[0]], [20])
        np.testing.assert_array_equal(
            associate.call_args_list[1].kwargs['detection_indices'], [7]
        )

    def test_lost_track_output_is_opt_in_and_decays_prediction_score(self):
        base_config = {
            'motion': {'enabled': False},
            'min_box_area': 1,
            'lost_output_max_age': 3,
            'lost_output_score_decay': 0.7,
            'lost_output_min_score': 0.1,
        }
        detections = np.array([
            [10.0, 10.0, 30.0, 30.0, 0.9],
        ])

        disabled = OCSORTTracker(base_config)
        disabled.update(detections)
        disabled.update(np.empty((0, 5)))
        # Legacy output behaviour is unchanged while the feature is disabled.
        self.assertEqual(len(disabled.get_outputs()), 1)

        tracker = OCSORTTracker({**base_config, 'output_lost_tracks': True})
        tracker.update(detections)
        tracker.update(np.empty((0, 5)))
        first_gap_output = tracker.get_outputs()
        self.assertEqual(len(first_gap_output), 1)
        self.assertEqual(first_gap_output[0].split(',')[6], '0.6')  # 0.9 * 0.7

        tracker.update(np.empty((0, 5)))
        self.assertEqual(tracker.get_outputs()[0].split(',')[6], '0.4')
        tracker.update(np.empty((0, 5)))
        self.assertEqual(tracker.get_outputs()[0].split(',')[6], '0.3')
        tracker.update(np.empty((0, 5)))
        self.assertEqual(tracker.get_outputs(), [])

    def test_lost_output_is_suppressed_when_prediction_leaves_frame(self):
        tracker = OCSORTTracker({
            'motion': {'enabled': False},
            'image_width': 100,
            'image_height': 100,
            'min_box_area': 1,
            'output_lost_tracks': True,
        })
        tracker.update(np.array([[10.0, 10.0, 30.0, 30.0, 0.9]]))
        tracker.update(np.empty((0, 5)))
        track = tracker.tracks[0]
        # Force this missed-frame prediction partly outside the left image edge.
        track.history.predict[track.current_frame].bbox = BBOX([5.0, 50.0, 20.0, 20.0])
        self.assertEqual(tracker.get_outputs(), [])

        tracker.config.lost_output_require_inside_frame = False
        self.assertEqual(len(tracker.get_outputs()), 1)

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

    def test_detection_file_path_includes_detector_name(self):
        args = SimpleNamespace(
            detections_dir='C:/Projects/.Detections',
            detector_name='YOLO26x',
            dataset='MOT20',
        )
        self.assertEqual(
            run_tracker.detection_file_path(args, 'MOT20-01'),
            os.path.join('C:/Projects/.Detections', 'YOLO26x', 'MOT20', 'MOT20-01.txt'),
        )

    def test_sequence_log_path_supports_directory_file_and_template(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_args = SimpleNamespace(
                log_path=os.path.join(directory, 'logs'), seqs=['seq-a', 'seq-b']
            )
            self.assertEqual(
                run_tracker.sequence_log_path(directory_args, 'seq-a'),
                os.path.join(directory, 'logs', 'seq-a.assoc.log'),
            )

            file_args = SimpleNamespace(
                log_path=os.path.join(directory, 'association.log'), seqs=['seq-a', 'seq-b']
            )
            self.assertEqual(
                run_tracker.sequence_log_path(file_args, 'seq-b'),
                os.path.join(directory, 'association-seq-b.log'),
            )

            template_args = SimpleNamespace(
                log_path=os.path.join(directory, '{seq}.log'), seqs=['seq-a', 'seq-b']
            )
            self.assertEqual(
                run_tracker.sequence_log_path(template_args, 'seq-b'),
                os.path.join(directory, 'seq-b.log'),
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
