"""
Service Dependency Impact — Meta AI-Enabled Round (Confirmed Pool #9)
=====================================================================

Problem:
    Given a directed service dependency graph where an edge A -> B means
    "service A depends on service B", determine which services are impacted
    when one or more services fail.

    A service is "impacted" if it transitively depends on a failed service.
    In other words: find all services that have a failed service as an
    ancestor in the dependency graph.

    Example:
        A -> B -> D
        C -> B
        E -> F

        If B fails: A and C are impacted (they depend on B directly or transitively).
        If D fails: B, A, C are all impacted.

Checkpoint structure (mirrors actual Meta AI-enabled interview):
    Checkpoint 1: Build the dependency graph (and its reverse for efficient lookup)
    Checkpoint 2: find_impacted(failed) — all services depending on any failed service
    Checkpoint 3: impact_score(service) — count of unique services impacted if this one fails
                  Useful for identifying high-blast-radius services for on-call prioritization

Key insight:
    Build the REVERSE graph (dependent -> depended-on becomes dependee -> dependents).
    Then BFS/DFS from failed nodes in the reverse graph finds all impacted services.

Complexity:
    build_graph:    Time O(E), Space O(V + E)
    find_impacted:  Time O(V + E), Space O(V)
    impact_score:   Time O(V * (V + E)), Space O(V)
"""

from collections import deque
from typing import Dict, List, Optional, Set


class ServiceGraph:
    """Directed dependency graph: edge (A -> B) means A depends on B.

    Args:
        services: Optional list of service names to pre-register.
    """

    def __init__(self, services: Optional[List[str]] = None) -> None:
        # depends_on[A] = set of services A depends on (outgoing edges)
        self._depends_on: Dict[str, Set[str]] = {}
        # dependents[B] = set of services that depend on B (incoming edges, reverse graph)
        self._dependents: Dict[str, Set[str]] = {}

        for s in (services or []):
            self._register(s)

    def _register(self, service: str) -> None:
        """Ensure a service exists in the graph with empty adjacency sets."""
        if service not in self._depends_on:
            self._depends_on[service] = set()
            self._dependents[service] = set()

    def add_dependency(self, service: str, depends_on: str) -> None:
        """Add edge: service depends on depends_on.

        Args:
            service:    The service that has the dependency.
            depends_on: The service being depended upon.

        Raises:
            ValueError: If service == depends_on (self-dependency).
        """
        if service == depends_on:
            raise ValueError(f"Service cannot depend on itself: {service!r}")
        self._register(service)
        self._register(depends_on)
        self._depends_on[service].add(depends_on)
        self._dependents[depends_on].add(service)

    def get_services(self) -> Set[str]:
        """Return all registered service names."""
        return set(self._depends_on.keys())

    # ------------------------------------------------------------------
    # Checkpoint 2: Impact analysis
    # ------------------------------------------------------------------

    def find_impacted(self, failed: "str | List[str]") -> Set[str]:
        """Find all services impacted by one or more failed services.

        A service is impacted if it transitively depends on any failed service.
        Uses BFS on the reverse graph (dependents direction).

        Args:
            failed: Single service name or list of failed service names.

        Returns:
            Set of impacted service names, NOT including the failed services themselves.

        Raises:
            KeyError: If any failed service is not in the graph.

        Complexity:
            Time:  O(V + E)
            Space: O(V)
        """
        if isinstance(failed, str):
            failed = [failed]

        impacted: Set[str] = set()
        queue: deque = deque()

        for svc in failed:
            if svc not in self._dependents:
                raise KeyError(f"Service {svc!r} not in graph")
            # Seed BFS with direct dependents of each failed service
            for dep in self._dependents[svc]:
                if dep not in impacted:
                    impacted.add(dep)
                    queue.append(dep)

        # BFS: propagate impact through the dependency chain
        while queue:
            svc = queue.popleft()
            for upstream in self._dependents.get(svc, set()):
                if upstream not in impacted:
                    impacted.add(upstream)
                    queue.append(upstream)

        return impacted

    # ------------------------------------------------------------------
    # Checkpoint 3: Impact score (blast radius)
    # ------------------------------------------------------------------

    def impact_score(self, service: str) -> int:
        """Count how many services would be impacted if this service fails.

        Args:
            service: The service whose failure to hypothesize.

        Returns:
            Number of impacted services (not counting the service itself).

        Raises:
            KeyError: If service not in graph.

        Complexity:
            Time:  O(V + E)
            Space: O(V)
        """
        return len(self.find_impacted(service))

    def top_blast_radius(self, top_k: int = 5) -> List[tuple[str, int]]:
        """Return top-k services sorted by blast radius (most impactful first).

        Args:
            top_k: Number of top services to return.

        Returns:
            List of (service_name, impact_count) tuples, descending by count.

        Complexity:
            Time:  O(V * (V + E) + V log V)
            Space: O(V)
        """
        scores = [(svc, self.impact_score(svc)) for svc in self.get_services()]
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Build test graph:
    # api_gateway -> auth_service -> user_db
    # api_gateway -> data_service -> user_db
    # api_gateway -> data_service -> cache
    # mobile_app  -> api_gateway
    graph = ServiceGraph()
    graph.add_dependency("api_gateway", "auth_service")
    graph.add_dependency("api_gateway", "data_service")
    graph.add_dependency("auth_service", "user_db")
    graph.add_dependency("data_service", "user_db")
    graph.add_dependency("data_service", "cache")
    graph.add_dependency("mobile_app", "api_gateway")

    # user_db failure: auth_service, data_service, api_gateway, mobile_app all impacted
    impacted_db = graph.find_impacted("user_db")
    assert "auth_service" in impacted_db
    assert "data_service" in impacted_db
    assert "api_gateway" in impacted_db
    assert "mobile_app" in impacted_db
    assert "user_db" not in impacted_db      # failed service not in result
    assert "cache" not in impacted_db        # cache does not depend on user_db

    # cache failure: only data_service, api_gateway, mobile_app impacted
    impacted_cache = graph.find_impacted("cache")
    assert "data_service" in impacted_cache
    assert "api_gateway" in impacted_cache
    assert "mobile_app" in impacted_cache
    assert "auth_service" not in impacted_cache

    # mobile_app failure: no dependents -> nothing impacted
    assert graph.find_impacted("mobile_app") == set()

    # Multiple failures
    impacted_multi = graph.find_impacted(["auth_service", "cache"])
    assert "api_gateway" in impacted_multi
    assert "data_service" in impacted_multi
    assert "mobile_app" in impacted_multi

    # Impact scores
    assert graph.impact_score("user_db") == 4   # auth, data, api, mobile
    assert graph.impact_score("cache") == 3     # data, api, mobile
    assert graph.impact_score("mobile_app") == 0

    # Top blast radius
    top = graph.top_blast_radius(top_k=3)
    assert top[0][0] == "user_db", f"user_db should have highest blast radius: {top}"
    assert top[0][1] == 4

    # Self-dependency error
    try:
        graph.add_dependency("x", "x")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    print("  service_dependency_impact: all tests passed")
    print(f"    Top blast radius: {graph.top_blast_radius(top_k=3)}")


if __name__ == "__main__":
    _test()
