"""
Word Ladder (LeetCode 127) — Hard
===================================

Problem:
    Given beginWord, endWord, and wordList, find the length of the shortest
    transformation sequence from beginWord to endWord where:
    - Each transformation changes exactly one letter.
    - Each intermediate word must exist in wordList.
    Return 0 if no such sequence exists.

Edge cases:
    - endWord not in wordList → 0
    - beginWord == endWord → 1 (count includes begin)
    - Empty wordList → 0
    - Single-character words

Approach:
    BFS from beginWord. Each level = one transformation step.
    Optimization: build a pattern graph offline (avoids O(W^2) neighbor search).
        Pattern "*ot" maps to {hot, dot, lot}; "h*t" maps to {hot, hit}.
    Then BFS on patterns, not word pairs.

    This is the approach Uber coding rounds have historically tested
    (graph traversal framed as a word transformation problem).

Complexity:
    Time:  O(M^2 * N) where M = word length, N = wordList size
           (M patterns per word, M pattern lookups per BFS expansion)
    Space: O(M^2 * N) for the pattern dictionary
"""

from collections import defaultdict, deque
from typing import List, Optional


def ladder_length(begin_word: str, end_word: str, word_list: List[str]) -> int:
    """Find shortest transformation sequence length.

    Args:
        begin_word: Starting word.
        end_word:   Target word.
        word_list:  List of valid intermediate words.

    Returns:
        Length of shortest sequence (including begin and end), or 0 if impossible.

    Complexity:
        Time:  O(M^2 * N)
        Space: O(M^2 * N)
    """
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    M = len(begin_word)

    # Build pattern → [words] adjacency map
    pattern_map: dict = defaultdict(list)
    for word in [begin_word] + word_list:
        for i in range(M):
            pattern = word[:i] + "*" + word[i + 1:]
            pattern_map[pattern].append(word)

    visited: set = {begin_word}
    queue: deque = deque([(begin_word, 1)])

    while queue:
        word, depth = queue.popleft()

        for i in range(M):
            pattern = word[:i] + "*" + word[i + 1:]
            for neighbor in pattern_map[pattern]:
                if neighbor == end_word:
                    return depth + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

    return 0


# ---------------------------------------------------------------------------
# Bidirectional BFS variant (interview follow-up — reduces search space)
# ---------------------------------------------------------------------------

def ladder_length_bidirectional(
    begin_word: str, end_word: str, word_list: List[str]
) -> int:
    """Bidirectional BFS — reduces average time complexity significantly.

    Expand from whichever frontier is smaller at each step.

    Complexity:
        Time:  O(M^2 * N) worst case; much faster in practice
        Space: O(M^2 * N)
    """
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    M = len(begin_word)
    pattern_map: dict = defaultdict(list)
    for word in word_list + [begin_word]:
        for i in range(M):
            pattern_map[word[:i] + "*" + word[i + 1:]].append(word)

    front = {begin_word: 1}
    back = {end_word: 1}
    visited_front = {begin_word}
    visited_back = {end_word}

    def expand(frontier: dict, visited_own: set, visited_other: set) -> Optional[int]:
        """Expand frontier by one level. Return depth if paths meet."""
        next_frontier: dict = {}
        for word, depth in frontier.items():
            for i in range(M):
                pattern = word[:i] + "*" + word[i + 1:]
                for neighbor in pattern_map[pattern]:
                    if neighbor in visited_other:
                        return depth + visited_other[neighbor]
                    if neighbor not in visited_own:
                        visited_own.add(neighbor)
                        next_frontier[neighbor] = depth + 1
        frontier.clear()
        frontier.update(next_frontier)
        return None

    while front and back:
        # Expand the smaller frontier
        if len(front) <= len(back):
            result = expand(front, visited_front, back)
        else:
            result = expand(back, visited_back, front)
        if result is not None:
            return result

    return 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    word_list_1 = ["hot", "dot", "dog", "lot", "log", "cog"]
    assert ladder_length("hit", "cog", word_list_1) == 5
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0  # cog missing
    assert ladder_length("a", "c", ["a", "b", "c"]) == 2

    # Bidirectional variant should give same result
    assert ladder_length_bidirectional("hit", "cog", word_list_1) == 5
    assert ladder_length_bidirectional("hit", "cog", ["hot", "dot"]) == 0

    print("  word_ladder: all tests passed")


if __name__ == "__main__":
    _test()
