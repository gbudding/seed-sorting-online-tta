"""
drift_sequences.py
------------------
Generates per-batch drift schedules for the streaming TTA experiments.

A drift sequence is a list of length `n_batches` where entry i is the
corruption configuration for batch i.  Each entry is either:

  - None            → clean baseline (no corruption applied)
  - (severity:int)  → severity level in {1, 2, 3, 4, 5}, with the
                      corruption *type* fixed for the whole sequence

The corruption type (one of 'brightness', 'blur', 'noise') is fixed per
run because the proposal isolates one corruption type per drift sequence
(see "Drift Sequences" section).  Combined corruptions are deferred to
future work.

Two scenarios are implemented:

1. `gradual_degradation`  — 300 batches, six contiguous 50-batch phases:
     batches   1–50  : clean
     batches  51–100 : severity 1
     batches 101–150 : severity 2
     batches 151–200 : severity 3
     batches 201–250 : severity 4
     batches 251–300 : severity 5

2. `degradation_recovery` — 300 batches, symmetric up/down ramp:
     batches   1–30  : clean (24-batch buffer + ramp-in)
     batches  31–150 : up-ramp,  five 24-batch phases, severities 1..5
     batches 151–270 : down-ramp, five 24-batch phases, severities 5..1
     batches 271–300 : clean (recovery / "post-fix" period)

Both scenarios deliberately reach all five severity levels so that a
single Dstream pass per (corruption × scenario × method × seed) covers
the full severity range relevant to the offline H1/H3 conditions.

Notes on implementation
-----------------------
- The scenarios are described purely in terms of (batch_index → severity).
  The actual image corruption is applied at evaluation time by combining
  this schedule with the corruption type and the existing severity tables
  in src/data/corruptions.py.

- Phase boundaries are *fixed* across runs.  Only the *order of images
  within each severity phase* is varied between seeds, per proposal
  (Procedure section).

- Total batch count and per-phase boundaries follow the proposal exactly;
  any change here changes the scientific commitment and should be
  flagged in the report.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence


# ------------------------------------------------------------------
# Schedule data structure
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BatchPlan:
    """Plan for a single batch.

    Attributes
    ----------
    batch_index : 0-based batch position in the sequence.
    severity    : 0 for clean, otherwise 1..5.  (0 means the corruption
                  is NOT applied for this batch — it is the clean
                  baseline.)
    """
    batch_index : int
    severity    : int

    @property
    def is_clean(self) -> bool:
        return self.severity == 0


@dataclass(frozen=True)
class DriftSchedule:
    """A full drift trajectory: name + ordered list of BatchPlans."""
    name        : str
    plans       : List[BatchPlan]
    n_batches   : int
    phase_table : List[tuple]   # list of (start_batch_inclusive, end_batch_inclusive, severity)

    def severities(self) -> List[int]:
        """Convenience: list of severity values per batch."""
        return [p.severity for p in self.plans]


# ------------------------------------------------------------------
# Scenario builders
# ------------------------------------------------------------------

def gradual_degradation(n_batches: int = 300) -> DriftSchedule:
    """
    Build the gradual degradation schedule.

    Six contiguous phases of equal length: clean → sev1 → sev2 → sev3 → sev4 → sev5.

    The default n_batches=300 yields the canonical 50-batch phases described
    in the proposal.  Other values are accepted for unit tests and timing
    benchmarks; the schedule scales proportionally and rounds phase
    boundaries to the nearest integer.
    """
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")

    # Six phases of equal length: severities (0, 1, 2, 3, 4, 5)
    phases = []
    severities = [0, 1, 2, 3, 4, 5]
    n_phases   = len(severities)
    base       = n_batches // n_phases
    remainder  = n_batches - base * n_phases   # distribute extras to leftmost phases

    plans = []
    cursor = 0
    for i, sev in enumerate(severities):
        length = base + (1 if i < remainder else 0)
        start  = cursor
        end    = cursor + length - 1   # inclusive
        phases.append((start, end, sev))
        for b in range(start, end + 1):
            plans.append(BatchPlan(batch_index=b, severity=sev))
        cursor = end + 1

    return DriftSchedule(
        name="gradual_degradation",
        plans=plans,
        n_batches=n_batches,
        phase_table=phases,
    )


def degradation_recovery(n_batches: int = 300) -> DriftSchedule:
    """
    Build the degradation-and-recovery schedule.

    Structure (default n_batches=300):
      30 batches clean, 5×24 up-ramp (sev 1→5), 5×24 down-ramp (sev 5→1), 30 clean.

    Severity 5 occupies the contiguous block at the boundary between
    up-ramp and down-ramp (24 batches at end of up-ramp + 24 at start
    of down-ramp = 48 batches at peak).

    For non-default n_batches the same proportional structure is applied:
    10% clean / 40% up-ramp / 40% down-ramp / 10% clean, with each
    sub-ramp split equally into five severity phases.
    """
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    if n_batches < 20:
        raise ValueError("n_batches must be >= 20 for a meaningful recovery scenario")

    # Proportions: 10 / 40 / 40 / 10  (clean / up / down / clean)
    n_clean_pre    = max(1, round(0.10 * n_batches))
    n_clean_post   = max(1, round(0.10 * n_batches))
    n_up           = max(5, round(0.40 * n_batches))
    n_down         = n_batches - n_clean_pre - n_clean_post - n_up
    if n_down < 5:
        # Fall back to equal up/down split if rounding misallocated
        n_up = (n_batches - n_clean_pre - n_clean_post) // 2
        n_down = n_batches - n_clean_pre - n_clean_post - n_up

    # Split up-ramp and down-ramp into 5 phases each
    def split_into_five(total: int) -> List[int]:
        base = total // 5
        rem  = total - base * 5
        return [base + (1 if i < rem else 0) for i in range(5)]

    up_lengths   = split_into_five(n_up)
    down_lengths = split_into_five(n_down)

    plans   = []
    phases  = []
    cursor  = 0

    def add_phase(length: int, severity: int) -> None:
        nonlocal cursor
        if length <= 0:
            return
        start = cursor
        end   = cursor + length - 1
        phases.append((start, end, severity))
        for b in range(start, end + 1):
            plans.append(BatchPlan(batch_index=b, severity=severity))
        cursor = end + 1

    add_phase(n_clean_pre, 0)
    for sev, length in zip([1, 2, 3, 4, 5], up_lengths):
        add_phase(length, sev)
    for sev, length in zip([5, 4, 3, 2, 1], down_lengths):
        add_phase(length, sev)
    add_phase(n_clean_post, 0)

    # Sanity: phases cover exactly [0, n_batches)
    assert cursor == n_batches, (
        f"degradation_recovery built {cursor} batches, expected {n_batches}")

    return DriftSchedule(
        name="degradation_recovery",
        plans=plans,
        n_batches=n_batches,
        phase_table=phases,
    )


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

SCENARIO_BUILDERS = {
    "gradual_degradation":  gradual_degradation,
    "degradation_recovery": degradation_recovery,
}


def build_schedule(name: str, n_batches: int = 300) -> DriftSchedule:
    """Builds a drift schedule by name.  Raises KeyError if unknown."""
    if name not in SCENARIO_BUILDERS:
        raise KeyError(
            f"Unknown drift scenario '{name}'. "
            f"Choose from: {list(SCENARIO_BUILDERS)}")
    return SCENARIO_BUILDERS[name](n_batches=n_batches)


# ------------------------------------------------------------------
# CLI: print a schedule for visual inspection
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print a drift schedule for inspection.")
    parser.add_argument("scenario",
                        choices=list(SCENARIO_BUILDERS.keys()))
    parser.add_argument("--n-batches", type=int, default=300)
    args = parser.parse_args()

    sched = build_schedule(args.scenario, n_batches=args.n_batches)

    print(f"Schedule: {sched.name}  ({sched.n_batches} batches)\n")
    print(f"{'phase':>5}  {'start':>5}  {'end':>5}  {'len':>4}  severity")
    for i, (s, e, sev) in enumerate(sched.phase_table):
        kind = "clean" if sev == 0 else f"sev{sev}"
        print(f"{i:>5}  {s:>5}  {e:>5}  {e-s+1:>4}  {kind}")

    print(f"\nTotal phases: {len(sched.phase_table)}")
    print(f"Total batches: {len(sched.plans)}")
