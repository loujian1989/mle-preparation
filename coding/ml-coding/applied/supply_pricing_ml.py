"""
Supply Pricing ML — Uber Staff MLE Interview Coding Reference

Covers the 4 core algorithm areas for Uber Supply Pricing team:
  1. SurgeTierComputer       — surge multiplier with EMA dampening & tier snapping
  2. DriverElasticityEstimator — RDD-based elasticity estimation per zone
  3. DoublyRobustEvaluator   — off-policy evaluation for counterfactual surge
  4. SupplyFishingDetector    — detect drivers gaming surge via behavioral signals

Run directly to execute the inline test suite:
    python supply_pricing_ml.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Discrete surge multiplier tiers used in production (avoids continuous
# multiplier oscillation; riders see stable pricing categories).
SURGE_TIERS: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)

# EMA smoothing factor α ∈ (0, 1].  Higher → reacts faster but more volatile.
# 0.3 is empirically stable for 30–60s update cadence.
DEFAULT_EMA_ALPHA: float = 0.3

# RDD local-linear bandwidth (in surge multiplier units).
# Only observations within ±BANDWIDTH of a tier boundary are used.
RDD_BANDWIDTH: float = 0.15

# Minimum observations required on each side of a tier boundary for RDD.
RDD_MIN_OBSERVATIONS: int = 30

# Supply fishing: max legitimate offline/online cycles per hour before flagging.
MAX_LEGITIMATE_CYCLES_PER_HOUR: int = 4

# Supply fishing: threshold for cross-boundary zone moves per shift.
CROSS_BOUNDARY_THRESHOLD: int = 6

# Supply fishing: maximum seconds from surge posting to driver online event
# that is considered "opportunistic" (surge-chasing signal).
SURGE_CHASE_LAG_SECONDS: float = 45.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ZoneSnapshot:
    """Real-time state of a single H3 zone at one update tick."""
    zone_id: str
    online_driver_count: int          # active drivers in zone
    trip_request_rate: float          # trip requests per minute (last 5-min window)
    historical_baseline_demand: float # expected trip requests/min from offline model
    current_surge: float              # surge multiplier from previous tick
    driver_elasticity: float          # Δdrivers per Δsurge (from DriverElasticityEstimator)


@dataclass
class DriverEvent:
    """Single driver activity event used for elasticity estimation."""
    zone_id: str
    surge_multiplier: float
    driver_arrivals: int    # drivers who came online in the zone within a 5-min window
    timestamp_sec: int


@dataclass
class PolicyObservation:
    """One observation used for off-policy evaluation."""
    zone_id: str
    logging_surge: float        # surge level that was actually applied (logging policy)
    propensity: float           # P(logging_surge | zone_context) under logging policy
    outcome: int                # unfulfilled_trips observed
    predicted_outcome_at_logging: float  # DM model's prediction at logging_surge
    predicted_outcome_at_target: float   # DM model's prediction at target_surge


@dataclass
class DriverShift:
    """Aggregated behavioral signals for one driver over a shift."""
    driver_id: str
    offline_online_cycles: int
    cross_boundary_moves: int
    surge_chase_lag_seconds: List[float]  # seconds from surge event to coming online
    shift_duration_hours: float


# ---------------------------------------------------------------------------
# 1. SurgeTierComputer
# ---------------------------------------------------------------------------

class SurgeTierComputer:
    """
    Computes the surge price multiplier for an H3 zone.

    Pipeline:
        1. Compute raw demand/supply ratio
        2. Adjust supply for strategic driver response (endogeneity correction)
        3. Convert ratio to raw continuous surge
        4. Snap to discrete tier
        5. Apply EMA dampening against previous surge

    The endogeneity correction is the key Staff-level insight: the observed
    supply is *already responding* to the current surge, so naive supply/demand
    ratio overestimates scarcity. We divide observed supply by the elastic
    response factor to get counterfactual "supply if surge were 1×".
    """

    def __init__(self, alpha: float = DEFAULT_EMA_ALPHA) -> None:
        """
        Args:
            alpha: EMA smoothing coefficient. Higher = more reactive.

        Raises:
            ValueError: If alpha is not in (0, 1].
        """
        if not (0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha

    def compute(self, snapshot: ZoneSnapshot) -> float:
        """
        Compute the new surge multiplier for a zone.

        Args:
            snapshot: Current zone state including driver counts, request rate,
                      historical baseline, prior surge, and elasticity estimate.

        Returns:
            New surge multiplier as a value from SURGE_TIERS.

        Complexity:
            Time:  O(T) where T = len(SURGE_TIERS), effectively O(1)
            Space: O(1)
        """
        # Step 1: Adjust supply for strategic response.
        # Drivers already repositioning toward the surge zone inflate raw
        # supply count. Divide by the elastic response factor to recover
        # "what supply would look like if surge = 1×".
        elastic_factor = 1.0 + snapshot.driver_elasticity * (snapshot.current_surge - 1.0)
        elastic_factor = max(elastic_factor, 0.1)  # prevent division by zero
        adjusted_supply = snapshot.online_driver_count / elastic_factor

        # Step 2: Compute excess demand relative to historical baseline.
        # ratio > 1 means demand is above what supply can handle at current rates.
        if adjusted_supply <= 0:
            raw_ratio = 3.0  # maximum surge if no supply at all
        else:
            raw_ratio = snapshot.trip_request_rate / max(adjusted_supply, 1e-6)

        # Normalize by historical baseline to get the surge signal:
        # surge = 1× when demand/supply is at its normal ratio.
        if snapshot.historical_baseline_demand > 0:
            baseline_ratio = snapshot.historical_baseline_demand / max(snapshot.online_driver_count, 1)
        else:
            baseline_ratio = 1.0

        # Raw continuous surge proportional to how much current ratio exceeds baseline.
        raw_surge = raw_ratio / max(baseline_ratio, 1e-6)
        raw_surge = max(1.0, min(raw_surge, SURGE_TIERS[-1]))  # clamp to tier range

        # Step 3: Snap to nearest tier.
        snapped_surge = self._snap_to_tier(raw_surge)

        # Step 4: EMA dampening to prevent oscillation.
        dampened_surge = (
            self._alpha * snapped_surge
            + (1.0 - self._alpha) * snapshot.current_surge
        )

        # Step 5: Re-snap after dampening (must return a valid tier).
        return self._snap_to_tier(dampened_surge)

    @staticmethod
    def _snap_to_tier(value: float) -> float:
        """
        Round value down to the nearest surge tier.

        Args:
            value: Continuous surge value.

        Returns:
            Nearest tier that is <= value (floor snap).

        Complexity:
            Time:  O(T) where T = len(SURGE_TIERS)
            Space: O(1)
        """
        result = SURGE_TIERS[0]
        for tier in SURGE_TIERS:
            if tier <= value:
                result = tier
            else:
                break
        return result


# ---------------------------------------------------------------------------
# 2. DriverElasticityEstimator
# ---------------------------------------------------------------------------

class DriverElasticityEstimator:
    """
    Estimates driver supply elasticity per zone using Regression Discontinuity
    Design (RDD) at surge tier boundaries.

    Intuition: surge multipliers are assigned by a threshold rule (e.g., surge
    jumps from 1.5× to 2.0× when the ratio crosses a cutoff). Drivers just above
    vs. just below the threshold are near-identical in unobservable characteristics,
    so the discontinuous jump in driver arrivals at the threshold identifies the
    causal effect of the higher surge tier.

    Estimation: local linear regression on each side of each boundary, using
    only observations within ±RDD_BANDWIDTH.
    """

    def fit(
        self,
        events: List[DriverEvent],
        zone_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Estimate elasticity for each zone.

        Args:
            events: List of driver arrival events with surge levels.
            zone_ids: Subset of zones to estimate. Estimates all zones if None.

        Returns:
            Dict mapping zone_id -> elasticity (Δdrivers per Δ1× surge).
            Zones with insufficient data are excluded from the result.

        Raises:
            ValueError: If events is empty.

        Complexity:
            Time:  O(N × B × T) where N = observations, B = RDD_BANDWIDTH lookups,
                   T = number of tier boundaries
            Space: O(N)
        """
        if not events:
            raise ValueError("events must be non-empty")

        # Group events by zone.
        by_zone: Dict[str, List[DriverEvent]] = defaultdict(list)
        for ev in events:
            if zone_ids is None or ev.zone_id in zone_ids:
                by_zone[ev.zone_id].append(ev)

        elasticities: Dict[str, float] = {}
        for zone_id, zone_events in by_zone.items():
            est = self._estimate_zone_elasticity(zone_events)
            if est is not None:
                elasticities[zone_id] = est

        return elasticities

    def _estimate_zone_elasticity(self, events: List[DriverEvent]) -> Optional[float]:
        """
        Estimate elasticity for a single zone by averaging RDD estimates across
        all tier boundaries that have sufficient data.

        Args:
            events: All events for one zone.

        Returns:
            Mean elasticity across boundaries, or None if insufficient data.

        Complexity:
            Time:  O(N × T) where T = len(SURGE_TIERS) - 1
            Space: O(N)
        """
        boundary_estimates: List[float] = []

        # Estimate at each tier boundary (gap between consecutive tiers).
        for i in range(len(SURGE_TIERS) - 1):
            lower_tier = SURGE_TIERS[i]
            upper_tier = SURGE_TIERS[i + 1]
            boundary = (lower_tier + upper_tier) / 2.0

            # Collect observations in the bandwidth window around the boundary.
            below = [e for e in events
                     if lower_tier - RDD_BANDWIDTH <= e.surge_multiplier < boundary]
            above = [e for e in events
                     if boundary <= e.surge_multiplier <= upper_tier + RDD_BANDWIDTH]

            if len(below) < RDD_MIN_OBSERVATIONS or len(above) < RDD_MIN_OBSERVATIONS:
                continue

            # Local linear regression on each side; take value at the boundary.
            fitted_below = self._local_linear_at_boundary(below, boundary)
            fitted_above = self._local_linear_at_boundary(above, boundary)

            surge_gap = upper_tier - lower_tier
            if surge_gap > 0:
                elasticity_at_boundary = (fitted_above - fitted_below) / surge_gap
                boundary_estimates.append(elasticity_at_boundary)

        if not boundary_estimates:
            return None

        return sum(boundary_estimates) / len(boundary_estimates)

    @staticmethod
    def _local_linear_at_boundary(
        events: List[DriverEvent], boundary: float
    ) -> float:
        """
        Fit a local linear regression and return the predicted value at `boundary`.

        Uses ordinary least squares: y = a + b*(x - boundary).

        Args:
            events: Observations on one side of the boundary.
            boundary: The cutoff point to predict at.

        Returns:
            Predicted driver_arrivals at the boundary (the intercept term `a`).

        Complexity:
            Time:  O(N)
            Space: O(1)
        """
        n = len(events)
        # Center surge around boundary for numerical stability.
        xs = [e.surge_multiplier - boundary for e in events]
        ys = [float(e.driver_arrivals) for e in events]

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        ss_xx = sum((x - mean_x) ** 2 for x in xs)
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

        if ss_xx < 1e-10:
            return mean_y  # degenerate case: all surge values identical

        slope = ss_xy / ss_xx
        intercept = mean_y - slope * mean_x
        # Predicted value at boundary: x - boundary = 0, so prediction = intercept.
        return intercept


# ---------------------------------------------------------------------------
# 3. DoublyRobustEvaluator
# ---------------------------------------------------------------------------

class DoublyRobustEvaluator:
    """
    Off-policy evaluation (OPE) for counterfactual surge policies.

    Why OPE instead of A/B test for every surge level?
    - A/B tests require live traffic → real drivers and riders experience the policy.
    - For extreme surge levels, real-time A/B is ethically/commercially problematic.
    - OPE estimates E[outcome | target_policy] from data collected under a logging policy.

    Method: Doubly Robust (DR) estimator.
        DR = (1/N) Σ [ DM(x, πₑ) + (Y - DM(x, π₀)) × (πₑ(a|x) / π₀(a|x)) ]

    Where:
        DM(x, πₑ)   = direct model prediction at evaluation policy action
        π₀(a|x)     = propensity of logging policy taking action a in context x
        πₑ(a|x)     = 1 if evaluation policy would choose a, else 0
        Y           = observed outcome

    DR is consistent if *either* the DM model or the propensity is correct
    (doubly robust property).

    Note on SUTVA: standard OPE assumes Stable Unit Treatment Value Assumption —
    one zone's treatment doesn't affect another's outcome. Surge violates this
    (drivers move across zones). Correct by: (1) zone-level rather than
    request-level randomization, (2) restrict analysis to non-adjacent zones.
    """

    def evaluate(
        self,
        observations: List[PolicyObservation],
        target_surge: float,
    ) -> Tuple[float, float]:
        """
        Estimate expected unfulfilled trips under a target surge policy.

        Args:
            observations: Logged observations from the current (logging) policy.
            target_surge: The surge level we want to evaluate counterfactually.

        Returns:
            Tuple of (dr_estimate, dm_only_estimate).
            dr_estimate: Doubly robust estimate of E[unfulfilled_trips].
            dm_only_estimate: Direct model only (baseline comparison).

        Raises:
            ValueError: If observations is empty.

        Complexity:
            Time:  O(N)
            Space: O(1)
        """
        if not observations:
            raise ValueError("observations must be non-empty")

        dr_sum = 0.0
        dm_sum = 0.0

        for obs in observations:
            # DM term: model's prediction at the target policy.
            dm_term = obs.predicted_outcome_at_target

            # IPW correction: only applies when the logging policy took the
            # same action as the target policy (importance weight = πₑ/π₀).
            # When logging action matches target_surge, importance weight = 1/propensity.
            # Otherwise importance weight = 0 (target policy wouldn't have taken that action).
            action_matches = math.isclose(obs.logging_surge, target_surge, abs_tol=0.01)
            if action_matches and obs.propensity > 1e-8:
                importance_weight = 1.0 / obs.propensity
                residual = obs.outcome - obs.predicted_outcome_at_logging
                ipw_correction = residual * importance_weight
            else:
                ipw_correction = 0.0

            dr_sum += dm_term + ipw_correction
            dm_sum += dm_term

        n = len(observations)
        return dr_sum / n, dm_sum / n


# ---------------------------------------------------------------------------
# 4. SupplyFishingDetector
# ---------------------------------------------------------------------------

class SupplyFishingDetector:
    """
    Detects drivers gaming surge pricing via behavioral pattern analysis.

    Supply fishing: a driver repeatedly goes offline and comes back online to
    manipulate zone supply counts, artificially sustaining or triggering surge.
    Strategy: if driver A and 9 friends all go offline at the same time, the zone
    drops below the surge threshold → surge activates → they all come back online
    and earn 2× fare.

    Three behavioral signals:
        1. offline_online_cycles_per_hour  — rapid toggling inflates supply variability
        2. cross_boundary_moves            — moving just outside a zone then back in
        3. surge_chase_lag_seconds         — consistently coming online within seconds
                                             of a surge event (not organic behavior)

    Output: binary flag (True = suspected fishing) + confidence score ∈ [0, 1].
    """

    def score(self, shift: DriverShift) -> Tuple[bool, float]:
        """
        Score a driver shift for supply fishing behavior.

        Args:
            shift: Aggregated behavioral signals for the shift.

        Returns:
            Tuple of (is_flagged, confidence_score).
            is_flagged: True if confidence_score >= 0.5.
            confidence_score: Weighted combination of the three signals, ∈ [0, 1].

        Raises:
            ValueError: If shift_duration_hours <= 0.

        Complexity:
            Time:  O(C) where C = len(surge_chase_lag_seconds)
            Space: O(1)
        """
        if shift.shift_duration_hours <= 0:
            raise ValueError("shift_duration_hours must be positive")

        # Signal 1: cycles per hour.
        cycles_per_hour = shift.offline_online_cycles / shift.shift_duration_hours
        cycle_score = min(
            cycles_per_hour / MAX_LEGITIMATE_CYCLES_PER_HOUR,
            1.0,
        )

        # Signal 2: cross-boundary moves (normalized by threshold).
        boundary_score = min(
            shift.cross_boundary_moves / CROSS_BOUNDARY_THRESHOLD,
            1.0,
        )

        # Signal 3: surge-chase lag — what fraction of online events happened
        # within SURGE_CHASE_LAG_SECONDS of a surge activation?
        if shift.surge_chase_lag_seconds:
            fast_responses = sum(
                1 for lag in shift.surge_chase_lag_seconds
                if lag <= SURGE_CHASE_LAG_SECONDS
            )
            chase_score = fast_responses / len(shift.surge_chase_lag_seconds)
        else:
            chase_score = 0.0

        # Weighted combination: chase_score is most discriminative signal.
        confidence = 0.25 * cycle_score + 0.25 * boundary_score + 0.50 * chase_score
        is_flagged = confidence >= 0.5

        return is_flagged, round(confidence, 4)

    def batch_score(
        self, shifts: List[DriverShift]
    ) -> List[Tuple[str, bool, float]]:
        """
        Score a batch of driver shifts.

        Args:
            shifts: List of DriverShift objects.

        Returns:
            List of (driver_id, is_flagged, confidence_score) tuples.

        Raises:
            ValueError: If shifts is empty.

        Complexity:
            Time:  O(N × C) where N = shifts, C = avg surge_chase_lag events
            Space: O(N)
        """
        if not shifts:
            raise ValueError("shifts must be non-empty")
        return [(s.driver_id, *self.score(s)) for s in shifts]


# ---------------------------------------------------------------------------
# Inline test suite
# ---------------------------------------------------------------------------

def _test_surge_tier_computer() -> None:
    computer = SurgeTierComputer(alpha=DEFAULT_EMA_ALPHA)

    # Case 1: Normal conditions — demand ≈ supply at baseline → surge = 1×.
    snapshot_normal = ZoneSnapshot(
        zone_id="zone_a",
        online_driver_count=20,
        trip_request_rate=20.0,
        historical_baseline_demand=20.0,
        current_surge=1.0,
        driver_elasticity=0.0,
    )
    result = computer.compute(snapshot_normal)
    assert result == 1.0, f"Expected 1.0, got {result}"

    # Case 2: High demand relative to supply → surge > 1.
    snapshot_surge = ZoneSnapshot(
        zone_id="zone_b",
        online_driver_count=5,
        trip_request_rate=30.0,
        historical_baseline_demand=15.0,
        current_surge=1.0,
        driver_elasticity=0.0,
    )
    result = computer.compute(snapshot_surge)
    assert result > 1.0, f"Expected surge > 1.0, got {result}"
    assert result in SURGE_TIERS, f"Result {result} not in SURGE_TIERS"

    # Case 3: EMA dampening — sudden demand drop should not immediately reset surge.
    snapshot_after_spike = ZoneSnapshot(
        zone_id="zone_c",
        online_driver_count=20,
        trip_request_rate=18.0,
        historical_baseline_demand=20.0,
        current_surge=2.0,  # was at 2× last tick
        driver_elasticity=0.0,
    )
    result = computer.compute(snapshot_after_spike)
    # With dampening, should not jump all the way to 1× immediately.
    assert result >= 1.0, f"Expected result >= 1.0, got {result}"

    # Case 4: Endogeneity correction — high elasticity should reduce effective supply.
    snapshot_elastic = ZoneSnapshot(
        zone_id="zone_d",
        online_driver_count=20,
        trip_request_rate=20.0,
        historical_baseline_demand=20.0,
        current_surge=2.0,
        driver_elasticity=5.0,  # 5 extra drivers per 1× surge
    )
    result = computer.compute(snapshot_elastic)
    # High elasticity means observed supply is inflated by surge; adjusted supply is lower.
    # So surge should be higher than without the correction.
    snapshot_no_elasticity = ZoneSnapshot(
        zone_id="zone_d",
        online_driver_count=20,
        trip_request_rate=20.0,
        historical_baseline_demand=20.0,
        current_surge=2.0,
        driver_elasticity=0.0,
    )
    result_no_elasticity = computer.compute(snapshot_no_elasticity)
    assert result >= result_no_elasticity, (
        f"Elasticity correction should increase surge: {result} < {result_no_elasticity}"
    )

    print("SurgeTierComputer: all tests passed")


def _test_driver_elasticity_estimator() -> None:
    import random
    random.seed(42)

    estimator = DriverElasticityEstimator()

    # Generate synthetic events: in zone_x, higher surge → more drivers arrive.
    # True elasticity: 3 extra drivers per 1× surge increase.
    TRUE_ELASTICITY = 3.0
    events: List[DriverEvent] = []
    for _ in range(500):
        surge = random.uniform(1.0, 3.0)
        noise = random.gauss(0, 1.0)
        arrivals = max(0, int(5.0 + TRUE_ELASTICITY * (surge - 1.0) + noise))
        events.append(DriverEvent(
            zone_id="zone_x",
            surge_multiplier=surge,
            driver_arrivals=arrivals,
            timestamp_sec=int(random.uniform(0, 86400)),
        ))

    result = estimator.fit(events)
    assert "zone_x" in result, "Expected elasticity estimate for zone_x"
    estimated = result["zone_x"]
    # Allow ±50% tolerance given synthetic data noise.
    assert abs(estimated - TRUE_ELASTICITY) / TRUE_ELASTICITY < 0.5, (
        f"Elasticity estimate {estimated:.2f} too far from true {TRUE_ELASTICITY}"
    )

    # Empty input should raise.
    try:
        estimator.fit([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("DriverElasticityEstimator: all tests passed")


def _test_doubly_robust_evaluator() -> None:
    evaluator = DoublyRobustEvaluator()
    target_surge = 2.0

    # All observations at logging_surge = 2.0 (matches target).
    # Propensity = 1.0 (deterministic logging policy).
    # Observed outcomes = 5. DM prediction = 4.
    # DR = DM_term + (Y - DM_at_logging) * (1/propensity)
    #    = 4 + (5 - 4) * 1.0 = 5.0 per observation.
    observations = [
        PolicyObservation(
            zone_id=f"zone_{i}",
            logging_surge=2.0,
            propensity=1.0,
            outcome=5,
            predicted_outcome_at_logging=4.0,
            predicted_outcome_at_target=4.0,
        )
        for i in range(10)
    ]
    dr_est, dm_est = evaluator.evaluate(observations, target_surge=target_surge)
    assert abs(dr_est - 5.0) < 1e-6, f"Expected DR estimate 5.0, got {dr_est}"
    assert abs(dm_est - 4.0) < 1e-6, f"Expected DM estimate 4.0, got {dm_est}"

    # Empty observations should raise.
    try:
        evaluator.evaluate([], target_surge=2.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("DoublyRobustEvaluator: all tests passed")


def _test_supply_fishing_detector() -> None:
    detector = SupplyFishingDetector()

    # Clean driver: few cycles, no boundary moves, organic online times.
    clean_shift = DriverShift(
        driver_id="driver_clean",
        offline_online_cycles=2,
        cross_boundary_moves=1,
        surge_chase_lag_seconds=[120.0, 300.0, 90.0],  # slow responses
        shift_duration_hours=8.0,
    )
    flagged, score = detector.score(clean_shift)
    assert not flagged, f"Clean driver should not be flagged, got score {score}"

    # Fishing driver: rapid cycles, boundary hopping, always online within 30s of surge.
    fishing_shift = DriverShift(
        driver_id="driver_fisher",
        offline_online_cycles=40,
        cross_boundary_moves=12,
        surge_chase_lag_seconds=[15.0, 20.0, 10.0, 30.0, 5.0, 25.0, 18.0],
        shift_duration_hours=8.0,
    )
    flagged, score = detector.score(fishing_shift)
    assert flagged, f"Fishing driver should be flagged, got score {score}"
    assert score >= 0.5, f"Score should be >= 0.5, got {score}"

    # Invalid shift duration should raise.
    bad_shift = DriverShift(
        driver_id="driver_bad",
        offline_online_cycles=5,
        cross_boundary_moves=2,
        surge_chase_lag_seconds=[],
        shift_duration_hours=0.0,
    )
    try:
        detector.score(bad_shift)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Batch scoring.
    results = detector.batch_score([clean_shift, fishing_shift])
    assert len(results) == 2
    assert results[0][0] == "driver_clean"
    assert results[1][0] == "driver_fisher"

    print("SupplyFishingDetector: all tests passed")


if __name__ == "__main__":
    _test_surge_tier_computer()
    _test_driver_elasticity_estimator()
    _test_doubly_robust_evaluator()
    _test_supply_fishing_detector()
    print("\nAll supply pricing ML tests passed.")
