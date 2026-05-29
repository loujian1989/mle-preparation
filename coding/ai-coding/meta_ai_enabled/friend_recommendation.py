"""
Friend Recommendation System — Meta AI-Enabled Round (Confirmed Pool #4)
=========================================================================

Problem:
    Given a social graph, recommend potential friends to a user based on mutual
    connection count. A valid recommendation must satisfy:
        1. Target is not already a friend of the user
        2. Target is not the user themselves
        3. Target is reachable within 2 hops (friend-of-friend)

    Rank recommendations by number of mutual friends (descending).
    Ties broken alphabetically by user ID.

Checkpoint structure (mirrors actual Meta AI-enabled interview):
    Checkpoint 1: User class with friend relationship management
    Checkpoint 2: count_mutual_friends(user_a, user_b) function
    Checkpoint 3: recommend_friends(user, top_k) — ranked list

Key insight:
    Mutual friend count = |friends(user) ∩ friends(candidate)|
    Candidates = union of friends-of-friends minus existing friends minus self
    Set intersection is O(min(|A|, |B|)) with Python sets.

Complexity:
    recommend_friends: Time O(F^2) where F = average friend count
                       Space O(F) for candidate set
"""

from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Checkpoint 1: User and Graph model
# ---------------------------------------------------------------------------

class User:
    """Represents a user in the social graph.

    Args:
        user_id: Unique string identifier for this user.

    Raises:
        ValueError: If user_id is empty.
    """

    def __init__(self, user_id: str) -> None:
        if not user_id:
            raise ValueError("user_id must be non-empty")
        self.user_id = user_id
        self.friends: Set[str] = set()  # set of friend user_ids

    def add_friend(self, other: "User") -> None:
        """Add a bidirectional friendship.

        Args:
            other: The other User to befriend.
        """
        self.friends.add(other.user_id)
        other.friends.add(self.user_id)

    def remove_friend(self, other: "User") -> None:
        """Remove a bidirectional friendship if it exists.

        Args:
            other: The User to unfriend.
        """
        self.friends.discard(other.user_id)
        other.friends.discard(self.user_id)

    def __repr__(self) -> str:
        return f"User({self.user_id!r}, friends={sorted(self.friends)})"


class SocialGraph:
    """Collection of users with lookup by ID.

    Args:
        users: Optional list of User objects to initialize with.
    """

    def __init__(self, users: Optional[List[User]] = None) -> None:
        self._users: Dict[str, User] = {}
        for u in (users or []):
            self.add_user(u)

    def add_user(self, user: User) -> None:
        """Register a user in the graph.

        Args:
            user: User to add.

        Raises:
            ValueError: If user_id already exists.
        """
        if user.user_id in self._users:
            raise ValueError(f"User '{user.user_id}' already in graph")
        self._users[user.user_id] = user

    def get_user(self, user_id: str) -> User:
        """Retrieve user by ID.

        Args:
            user_id: Target user ID.

        Returns:
            Corresponding User object.

        Raises:
            KeyError: If user_id not in graph.
        """
        if user_id not in self._users:
            raise KeyError(f"User '{user_id}' not found")
        return self._users[user_id]

    # ------------------------------------------------------------------
    # Checkpoint 2: Mutual friend count
    # ------------------------------------------------------------------

    def count_mutual_friends(self, id_a: str, id_b: str) -> int:
        """Count friends shared between users a and b.

        Args:
            id_a: First user ID.
            id_b: Second user ID.

        Returns:
            Number of users that are friends with both a and b.

        Complexity:
            Time:  O(min(|friends_a|, |friends_b|))
            Space: O(1)
        """
        user_a = self.get_user(id_a)
        user_b = self.get_user(id_b)
        return len(user_a.friends & user_b.friends)

    # ------------------------------------------------------------------
    # Checkpoint 3: Ranked recommendations
    # ------------------------------------------------------------------

    def recommend_friends(self, user_id: str, top_k: int = 5) -> List[str]:
        """Return top-k friend recommendations for a user.

        Criteria:
            - Must be a friend-of-friend (within 2 hops)
            - Not already a direct friend
            - Not the user themselves
        Ranked by mutual friend count descending; ties broken alphabetically.

        Args:
            user_id: User to generate recommendations for.
            top_k:   Maximum number of recommendations to return.

        Returns:
            List of recommended user IDs, up to top_k.

        Raises:
            KeyError: If user_id not in graph.

        Complexity:
            Time:  O(F^2) where F = average friend count
            Space: O(F)
        """
        user = self.get_user(user_id)
        direct_friends = user.friends

        # Candidates: friends-of-friends not already connected
        candidates: Set[str] = set()
        for friend_id in direct_friends:
            friend = self.get_user(friend_id)
            for fof_id in friend.friends:
                if fof_id != user_id and fof_id not in direct_friends:
                    candidates.add(fof_id)

        # Score each candidate
        scored: List[Tuple[int, str]] = []
        for candidate_id in candidates:
            mutual = self.count_mutual_friends(user_id, candidate_id)
            scored.append((mutual, candidate_id))

        # Sort: most mutuals first, then alphabetical for ties
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [uid for _, uid in scored[:top_k]]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Build graph:
    # alice -- bob -- carol
    #       \-- dave -- eve
    #                \-- frank
    alice = User("alice")
    bob = User("bob")
    carol = User("carol")
    dave = User("dave")
    eve = User("eve")
    frank = User("frank")

    alice.add_friend(bob)
    alice.add_friend(dave)
    bob.add_friend(carol)
    dave.add_friend(eve)
    dave.add_friend(frank)

    graph = SocialGraph([alice, bob, carol, dave, eve, frank])

    # Mutual friends
    assert graph.count_mutual_friends("alice", "carol") == 1   # mutual: bob
    assert graph.count_mutual_friends("alice", "eve") == 1     # mutual: dave
    assert graph.count_mutual_friends("alice", "frank") == 1   # mutual: dave
    assert graph.count_mutual_friends("bob", "dave") == 1      # mutual: alice

    # Recommendations for alice
    recs = graph.recommend_friends("alice", top_k=5)
    assert "carol" in recs, f"carol should be recommended: {recs}"
    assert "eve" in recs, f"eve should be recommended: {recs}"
    assert "frank" in recs, f"frank should be recommended: {recs}"
    assert "bob" not in recs, "bob is already alice's friend"
    assert "dave" not in recs, "dave is already alice's friend"
    assert "alice" not in recs, "alice should not recommend herself"

    # Add another mutual friend to make carol rank higher
    alice2 = User("alice2")
    bob2 = User("bob2")
    carol2 = User("carol2")
    dave2 = User("dave2")
    alice2.add_friend(bob2)
    alice2.add_friend(dave2)
    bob2.add_friend(carol2)
    dave2.add_friend(carol2)  # carol2 now has 2 mutual friends with alice2
    graph2 = SocialGraph([alice2, bob2, carol2, dave2])

    recs2 = graph2.recommend_friends("alice2", top_k=3)
    assert recs2[0] == "carol2", f"carol2 (2 mutuals) should rank first: {recs2}"

    # Empty friends -> no recommendations
    loner = User("loner")
    graph.add_user(loner)
    assert graph.recommend_friends("loner") == []

    print("  friend_recommendation: all tests passed")


if __name__ == "__main__":
    _test()
