"""
Max Unique Characters Subset — Meta AI-Enabled Round (Confirmed Pool #2)
========================================================================

Problem:
    Given a list of words, find a subset such that no character appears in
    more than one word in the subset. Maximize the total unique character count.

    Example:
        words = ["ab", "cd", "abc", "d"]
        Best subset: ["abc", "d"]  -> 4 unique chars {a,b,c,d}
        (or ["ab", "cd"]           -> 4, also valid)

Checkpoint structure (mirrors actual Meta AI-enabled interview):
    Checkpoint 1: Naive backtracking — correct, O(2^N * N), too slow for N > 20
    Checkpoint 2: Bitmask DP — encode each word as 26-bit int; DP over combined masks
    Checkpoint 3: Path reconstruction — return the actual words, not just the count

Key insight (bitmask DP):
    word_mask = integer where bit j = 1 iff chr(ord('a')+j) in word
    Compatibility: mask_a & mask_b == 0  (no shared chars)
    DP: dp[combined_mask] = max unique chars achievable with exactly this char set
    Transition: dp[mask | wm] = max(dp[mask | wm], dp[mask] + popcount(wm))
                only when mask & wm == 0

    Deduplication: words with identical char sets (same mask) are interchangeable;
    keep one representative per mask — reduces search space to unique masks.

Complexity:
    Backtracking:  Time O(2^N * N)   Space O(N)
    Bitmask DP:    Time O(U^2)       Space O(U)
                   where U = number of unique word masks (typically << N)
"""

from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _word_to_mask(word: str) -> int:
    """Convert word to 26-bit integer bitmask of its unique lowercase characters.

    Args:
        word: Lowercase word.

    Returns:
        Bitmask integer; bit i is set iff chr(ord('a') + i) appears in word.

    Complexity:
        Time:  O(len(word))
        Space: O(1)
    """
    mask = 0
    for ch in word:
        mask |= 1 << (ord(ch) - ord("a"))
    return mask


# ---------------------------------------------------------------------------
# Checkpoint 1: Backtracking (correct, readable, slow)
# ---------------------------------------------------------------------------

def max_unique_chars_backtrack(words: List[str]) -> Tuple[int, List[str]]:
    """Find max unique-char subset via recursive backtracking.

    Args:
        words: List of lowercase words.

    Returns:
        (max_chars, best_subset) tuple where best_subset is the winning word list.

    Raises:
        ValueError: If words is empty.

    Complexity:
        Time:  O(2^N * N) — explore every compatible subset
        Space: O(N)       — recursion depth
    """
    if not words:
        raise ValueError("words must be non-empty")

    masks = [_word_to_mask(w) for w in words]
    best_chars: List[int] = [0]
    best_subset: List[List[str]] = [[]]

    def _backtrack(idx: int, combined: int, chosen: List[str]) -> None:
        bits = bin(combined).count("1")
        if bits > best_chars[0]:
            best_chars[0] = bits
            best_subset[0] = list(chosen)
        for i in range(idx, len(words)):
            if combined & masks[i] == 0:  # no char overlap
                chosen.append(words[i])
                _backtrack(i + 1, combined | masks[i], chosen)
                chosen.pop()

    _backtrack(0, 0, [])
    return best_chars[0], best_subset[0]


# ---------------------------------------------------------------------------
# Checkpoint 2 + 3: Bitmask DP with path reconstruction (scale-ready)
# ---------------------------------------------------------------------------

def max_unique_chars_bitmask(words: List[str]) -> Tuple[int, List[str]]:
    """Find max unique-char subset using bitmask DP.

    Deduplicates words that share the same char set, then DP over unique masks.

    Args:
        words: List of lowercase words.

    Returns:
        (max_chars, best_subset) tuple.

    Raises:
        ValueError: If words is empty.

    Complexity:
        Time:  O(U^2) where U = number of unique word masks
        Space: O(U)
    """
    if not words:
        raise ValueError("words must be non-empty")

    # Dedup: one representative per unique char set
    mask_to_word: Dict[int, str] = {}
    for w in words:
        m = _word_to_mask(w)
        if m not in mask_to_word:
            mask_to_word[m] = w

    unique_masks = list(mask_to_word.keys())

    # dp[combined_mask] = (max_unique_chars, [chosen_words])
    dp: Dict[int, Tuple[int, List[str]]] = {0: (0, [])}

    for wm in unique_masks:
        word_bits = bin(wm).count("1")
        # snapshot to avoid modifying dp while iterating
        for combined, (chars, chosen) in list(dp.items()):
            if combined & wm == 0:
                new_mask = combined | wm
                new_chars = chars + word_bits
                if new_mask not in dp or dp[new_mask][0] < new_chars:
                    dp[new_mask] = (new_chars, chosen + [mask_to_word[wm]])

    best_chars, best_words = max(dp.values(), key=lambda x: x[0])
    return best_chars, best_words


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Basic: mixed overlapping and non-overlapping words
    words1 = ["ab", "cd", "abc", "d"]
    c_bt, s_bt = max_unique_chars_backtrack(words1)
    c_bm, s_bm = max_unique_chars_bitmask(words1)
    assert c_bt == 4, f"Backtrack: expected 4, got {c_bt}"
    assert c_bm == 4, f"Bitmask: expected 4, got {c_bm}"

    # Single word
    assert max_unique_chars_backtrack(["abcde"])[0] == 5
    assert max_unique_chars_bitmask(["abcde"])[0] == 5

    # All words share char 'a' — best is any single word (2 chars each)
    words3 = ["ab", "ac", "ad"]
    assert max_unique_chars_backtrack(words3)[0] == 2
    assert max_unique_chars_bitmask(words3)[0] == 2

    # Larger case: consistency between both approaches
    words4 = ["ijk", "lmn", "op", "qrs", "ijklmnop"]
    c4_bt, _ = max_unique_chars_backtrack(words4)
    c4_bm, _ = max_unique_chars_bitmask(words4)
    assert c4_bt == c4_bm, f"Mismatch: backtrack={c4_bt} bitmask={c4_bm}"

    # Verify returned subset is actually valid (no shared chars)
    def _valid(subset: List[str]) -> bool:
        seen: set = set()
        for w in subset:
            chars = set(w)
            if chars & seen:
                return False
            seen |= chars
        return True

    for words in [words1, words3, words4]:
        _, subset_bm = max_unique_chars_bitmask(words)
        assert _valid(subset_bm), f"Invalid subset returned: {subset_bm}"

    print("  max_unique_chars: all tests passed")
    print(f"    words1 -> count={c_bm}, subset={s_bm}")


if __name__ == "__main__":
    _test()
