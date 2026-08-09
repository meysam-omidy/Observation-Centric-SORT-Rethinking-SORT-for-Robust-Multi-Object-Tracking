import unittest
from types import SimpleNamespace

import numpy as np

from ocsort import OCSORTTracker
from track import TrackHistoryItem
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


if __name__ == '__main__':
    unittest.main()
