"""
test_drift_sequences.py
-----------------------
Unit tests for the drift sequence generator.

Run: python -m pytest tests/test_drift_sequences.py -v
or:  python tests/test_drift_sequences.py
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.drift_sequences import (
    BatchPlan,
    DriftSchedule,
    gradual_degradation,
    degradation_recovery,
    build_schedule,
)


class TestGradualDegradation(unittest.TestCase):
    """Tests for the gradual degradation schedule."""

    def test_default_length_is_300(self):
        s = gradual_degradation()
        self.assertEqual(len(s.plans), 300)
        self.assertEqual(s.n_batches, 300)

    def test_default_phases_match_proposal(self):
        """Six contiguous 50-batch phases at default n_batches=300."""
        s = gradual_degradation(n_batches=300)
        expected = [
            (  0,  49, 0),  # clean
            ( 50,  99, 1),
            (100, 149, 2),
            (150, 199, 3),
            (200, 249, 4),
            (250, 299, 5),
        ]
        self.assertEqual(s.phase_table, expected)

    def test_severity_at_specific_batches(self):
        s = gradual_degradation(n_batches=300)
        # First batch clean
        self.assertEqual(s.plans[0].severity, 0)
        # Batch 50 (0-indexed) -> severity 1 (proposal: "severity 1 for batches 51-100")
        self.assertEqual(s.plans[50].severity, 1)
        # Last batch -> severity 5
        self.assertEqual(s.plans[299].severity, 5)

    def test_phases_cover_full_range_no_gaps(self):
        s = gradual_degradation(n_batches=300)
        # Walk phases: each must be contiguous with the previous
        for i in range(1, len(s.phase_table)):
            prev_end = s.phase_table[i-1][1]
            curr_start = s.phase_table[i][0]
            self.assertEqual(prev_end + 1, curr_start,
                             f"Gap between phase {i-1} and {i}")
        # First starts at 0, last ends at n_batches-1
        self.assertEqual(s.phase_table[0][0], 0)
        self.assertEqual(s.phase_table[-1][1], 299)

    def test_severities_strictly_non_decreasing(self):
        s = gradual_degradation()
        sevs = s.severities()
        for i in range(1, len(sevs)):
            self.assertGreaterEqual(sevs[i], sevs[i-1],
                                    "Gradual degradation must be monotonic")

    def test_alternative_n_batches_120(self):
        """Smaller schedules for unit testing should still cover all phases."""
        s = gradual_degradation(n_batches=120)
        self.assertEqual(len(s.plans), 120)
        # All six severities (0..5) should be present
        sevs_present = set(s.severities())
        self.assertEqual(sevs_present, {0, 1, 2, 3, 4, 5})

    def test_zero_or_negative_n_batches_raises(self):
        with self.assertRaises(ValueError):
            gradual_degradation(n_batches=0)
        with self.assertRaises(ValueError):
            gradual_degradation(n_batches=-5)


class TestDegradationRecovery(unittest.TestCase):
    """Tests for the degradation-and-recovery schedule."""

    def test_default_length_is_300(self):
        s = degradation_recovery()
        self.assertEqual(len(s.plans), 300)
        self.assertEqual(s.n_batches, 300)

    def test_starts_and_ends_clean(self):
        s = degradation_recovery(n_batches=300)
        self.assertEqual(s.plans[0].severity, 0)
        self.assertEqual(s.plans[-1].severity, 0)

    def test_peak_severity_in_middle(self):
        """Severity 5 must occur somewhere near the middle."""
        s = degradation_recovery(n_batches=300)
        sevs = s.severities()
        self.assertEqual(max(sevs), 5)
        peak_indices = [i for i, x in enumerate(sevs) if x == 5]
        # Peak should bracket batch 150 (50% mark)
        self.assertTrue(min(peak_indices) <= 150 <= max(peak_indices),
                        "Severity 5 should bracket the midpoint")

    def test_default_proportions_close_to_proposal(self):
        s = degradation_recovery(n_batches=300)
        # 10/40/40/10 split of 300 = 30/120/120/30
        # Phase table: clean(30) + 5 up + 5 down + clean(30) = 12 phases
        self.assertEqual(len(s.phase_table), 12)
        clean_pre  = s.phase_table[0]
        clean_post = s.phase_table[-1]
        self.assertEqual(clean_pre[2], 0)
        self.assertEqual(clean_post[2], 0)
        # Each clean buffer 30 batches at 300
        self.assertEqual(clean_pre[1] - clean_pre[0] + 1, 30)
        self.assertEqual(clean_post[1] - clean_post[0] + 1, 30)

    def test_up_ramp_severities_in_order(self):
        s = degradation_recovery(n_batches=300)
        # phases[1..5] should be sev 1..5
        for i, sev in enumerate([1, 2, 3, 4, 5], start=1):
            self.assertEqual(s.phase_table[i][2], sev,
                             f"Phase {i} should be sev {sev}")

    def test_down_ramp_severities_in_order(self):
        s = degradation_recovery(n_batches=300)
        # phases[6..10] should be sev 5..1
        for i, sev in enumerate([5, 4, 3, 2, 1], start=6):
            self.assertEqual(s.phase_table[i][2], sev,
                             f"Phase {i} should be sev {sev}")

    def test_no_gaps_or_overlaps(self):
        s = degradation_recovery(n_batches=300)
        for i in range(1, len(s.phase_table)):
            prev_end = s.phase_table[i-1][1]
            curr_start = s.phase_table[i][0]
            self.assertEqual(prev_end + 1, curr_start,
                             f"Gap between phase {i-1} and {i}")
        self.assertEqual(s.phase_table[0][0], 0)
        self.assertEqual(s.phase_table[-1][1], 299)

    def test_symmetry_of_up_and_down_ramps(self):
        """Up-ramp and down-ramp should have the same total length at default sizing."""
        s = degradation_recovery(n_batches=300)
        up   = sum(p[1] - p[0] + 1 for p in s.phase_table[1:6])
        down = sum(p[1] - p[0] + 1 for p in s.phase_table[6:11])
        self.assertEqual(up, down)


class TestBuildSchedule(unittest.TestCase):
    def test_dispatch_to_gradual(self):
        s = build_schedule("gradual_degradation")
        self.assertEqual(s.name, "gradual_degradation")

    def test_dispatch_to_recovery(self):
        s = build_schedule("degradation_recovery")
        self.assertEqual(s.name, "degradation_recovery")

    def test_unknown_scenario_raises(self):
        with self.assertRaises(KeyError):
            build_schedule("not_a_scenario")


class TestBatchPlan(unittest.TestCase):
    def test_is_clean(self):
        self.assertTrue(BatchPlan(0, 0).is_clean)
        self.assertFalse(BatchPlan(0, 1).is_clean)
        self.assertFalse(BatchPlan(0, 5).is_clean)


if __name__ == "__main__":
    unittest.main(verbosity=2)
