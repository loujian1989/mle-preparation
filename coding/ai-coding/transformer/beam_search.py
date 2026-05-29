"""
Beam Search — Transformer Component
=====================================

Meta ML coding probe: "Implement greedy decoding and beam search. When would
you prefer beam search over greedy? When would you prefer sampling?"

Concepts:
    Greedy Decoding: At each step, pick the single highest-probability token.
        - Fast: O(V * seq_len) per sequence
        - Suboptimal: local choices can close off globally better paths

    Beam Search: Maintain the top-k (beam_width) partial sequences at each step.
        - O(V * beam_width * seq_len) per sequence
        - Finds better sequences than greedy; used in machine translation, summarization
        - Tends to produce shorter, more generic output for language models

    Temperature Sampling: Sample from softmax(logits / T)
        - T < 1 sharpens distribution (more deterministic)
        - T > 1 flattens distribution (more creative/random)

    Top-k Sampling: Sample from top-k most probable tokens (nucleus subset)
    Top-p (Nucleus) Sampling: Sample from smallest set whose cumulative prob >= p

When to use:
    Beam search:    Fixed-output tasks (translation, ASR) — needs single best answer
    Greedy:         When latency matters and quality is acceptable
    Sampling:       Creative generation (stories, chat) — diversity valued over precision
    Temperature:    Control creativity vs coherence trade-off

Length penalty in beam search:
    Without it, beam search prefers shorter sequences (fewer log-prob multiplications).
    Length penalty: score / (seq_len ^ alpha), alpha typically 0.6–0.8

Complexity:
    Greedy:      Time O(V * seq_len)          Space O(seq_len)
    Beam search: Time O(V * B * seq_len)      Space O(B * seq_len)
    where V = vocab size, B = beam width
"""

import numpy as np
from typing import Callable, List, Optional, Tuple


LogProbFn = Callable[[List[int]], np.ndarray]
# Takes a partial sequence and returns log-probabilities over vocabulary (1D array)


EOS_TOKEN = 0   # token ID for end-of-sequence
BOS_TOKEN = 1   # token ID for beginning-of-sequence


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling.

    Args:
        logits:      1D array of unnormalized log-scores.
        temperature: Controls sharpness; must be > 0.

    Returns:
        Probability distribution over vocabulary.
    """
    scaled = logits / max(temperature, 1e-8)
    scaled -= scaled.max()
    probs = np.exp(scaled)
    return probs / probs.sum()


# ---------------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------------

def greedy_decode(
    log_prob_fn: LogProbFn,
    max_len: int = 20,
    bos_token: int = BOS_TOKEN,
    eos_token: int = EOS_TOKEN,
) -> List[int]:
    """Decode a sequence by always choosing the most probable next token.

    Args:
        log_prob_fn: Function mapping partial sequence -> log-probabilities over vocab.
        max_len:     Maximum tokens to generate (including EOS).
        bos_token:   Beginning-of-sequence token to seed generation.
        eos_token:   End-of-sequence token; decoding stops when generated.

    Returns:
        Generated token sequence (not including BOS, including EOS if generated).

    Complexity:
        Time:  O(V * seq_len)
        Space: O(seq_len)
    """
    tokens = [bos_token]

    for _ in range(max_len):
        log_probs = log_prob_fn(tokens)
        next_token = int(np.argmax(log_probs))
        tokens.append(next_token)
        if next_token == eos_token:
            break

    return tokens[1:]  # strip BOS


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def beam_search(
    log_prob_fn: LogProbFn,
    beam_width: int = 3,
    max_len: int = 20,
    bos_token: int = BOS_TOKEN,
    eos_token: int = EOS_TOKEN,
    length_penalty: float = 0.0,
) -> List[Tuple[List[int], float]]:
    """Decode using beam search, returning top-k completed hypotheses.

    Args:
        log_prob_fn:    Function mapping partial sequence -> log-probs over vocab.
        beam_width:     Number of beams to maintain (k in top-k).
        max_len:        Maximum tokens to generate per sequence.
        bos_token:      Seed token.
        eos_token:      Stopping token.
        length_penalty: Alpha for length normalization: score / (len ^ alpha).
                        0 = no penalty, 0.6–0.8 typical for NMT.

    Returns:
        List of (token_sequence, normalized_score) tuples, sorted by score descending.
        Best hypothesis is first.

    Raises:
        ValueError: If beam_width < 1.

    Complexity:
        Time:  O(V * beam_width * seq_len)
        Space: O(beam_width * seq_len)
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    # Each beam: (tokens, cumulative_log_prob)
    beams: List[Tuple[List[int], float]] = [([bos_token], 0.0)]
    completed: List[Tuple[List[int], float]] = []

    for _ in range(max_len):
        if not beams:
            break

        candidates: List[Tuple[List[int], float]] = []

        for tokens, score in beams:
            log_probs = log_prob_fn(tokens)

            # Expand: take top beam_width continuations
            top_indices = np.argsort(log_probs)[-beam_width:]

            for idx in top_indices:
                new_tokens = tokens + [int(idx)]
                new_score = score + float(log_probs[idx])

                if int(idx) == eos_token:
                    completed.append((new_tokens[1:], new_score))  # strip BOS
                else:
                    candidates.append((new_tokens, new_score))

        # Prune to top beam_width candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    # Add unfinished beams as completed
    for tokens, score in beams:
        completed.append((tokens[1:], score))  # strip BOS

    # Apply length penalty and sort
    def _normalized_score(seq: List[int], score: float) -> float:
        length = max(len(seq), 1)
        return score / (length ** length_penalty)

    completed.sort(key=lambda x: _normalized_score(x[0], x[1]), reverse=True)
    return completed


# ---------------------------------------------------------------------------
# Sampling-based decoding
# ---------------------------------------------------------------------------

def sample_decode(
    log_prob_fn: LogProbFn,
    max_len: int = 20,
    temperature: float = 1.0,
    top_k: int = 0,
    bos_token: int = BOS_TOKEN,
    eos_token: int = EOS_TOKEN,
    seed: Optional[int] = None,
) -> List[int]:
    """Decode by sampling from the distribution with optional top-k truncation.

    Args:
        log_prob_fn: Function mapping partial sequence -> log-probs over vocab.
        max_len:     Maximum tokens to generate.
        temperature: Controls randomness (< 1 sharper, > 1 flatter).
        top_k:       If > 0, restrict sampling to top-k tokens.
        bos_token:   Seed token.
        eos_token:   Stop token.
        seed:        RNG seed for reproducibility.

    Returns:
        Sampled token sequence (not including BOS).

    Raises:
        ValueError: If temperature <= 0.

    Complexity:
        Time:  O(V * seq_len)
        Space: O(seq_len)
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    rng = np.random.default_rng(seed)
    tokens = [bos_token]

    for _ in range(max_len):
        log_probs = log_prob_fn(tokens)
        probs = _softmax(log_probs, temperature)

        if top_k > 0:
            # Zero out all but top-k
            top_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_indices] = probs[top_indices]
            probs = mask / mask.sum()

        next_token = int(rng.choice(len(probs), p=probs))
        tokens.append(next_token)
        if next_token == eos_token:
            break

    return tokens[1:]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test() -> None:
    # Toy language model: a 5-token vocabulary (0=EOS, 1=BOS, 2='a', 3='b', 4='c')
    # Deterministic: always predicts token 2 ('a') then 3 ('b') then 0 (EOS)
    VOCAB_SIZE = 5
    _step_counter = [0]

    def toy_log_prob_fn(tokens: List[int]) -> np.ndarray:
        log_probs = np.full(VOCAB_SIZE, -10.0)
        step = len(tokens) - 1  # how many tokens generated after BOS
        if step == 0:
            log_probs[2] = 0.0    # predict 'a'
        elif step == 1:
            log_probs[3] = 0.0    # predict 'b'
        else:
            log_probs[EOS_TOKEN] = 0.0  # predict EOS
        return log_probs

    # Greedy: should produce [2, 3, 0] = ['a', 'b', EOS]
    greedy_out = greedy_decode(toy_log_prob_fn, max_len=10)
    assert greedy_out == [2, 3, 0], f"Greedy output: {greedy_out}"

    # Beam search: best hypothesis should also be [2, 3, 0]
    beam_out = beam_search(toy_log_prob_fn, beam_width=3, max_len=10)
    assert beam_out[0][0] == [2, 3, 0], f"Beam best: {beam_out[0][0]}"

    # Beam search returns multiple hypotheses (up to beam_width)
    assert len(beam_out) >= 1

    # Sampling with temperature=0.01 (near-greedy): should mostly produce 'a','b','EOS'
    sample_out = sample_decode(toy_log_prob_fn, temperature=0.01, seed=42)
    assert sample_out == [2, 3, 0], f"Near-greedy sample: {sample_out}"

    # Beam search with length penalty: scores should be normalized
    beam_lp = beam_search(toy_log_prob_fn, beam_width=2, max_len=10, length_penalty=0.6)
    assert len(beam_lp) >= 1

    # ValueError on bad inputs
    try:
        beam_search(toy_log_prob_fn, beam_width=0)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    try:
        sample_decode(toy_log_prob_fn, temperature=-1.0)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # Comparative: verify beam search finds higher-scoring sequence than greedy
    # when greedy makes a locally suboptimal choice
    # Vocabulary: 0=EOS, 1=BOS, 2='x' (high step1, low step2), 3='y' (low step1, high step2+EOS)
    def suboptimal_fn(tokens: List[int]) -> np.ndarray:
        log_probs = np.full(5, -100.0)
        step = len(tokens) - 1
        if step == 0:
            log_probs[2] = -0.1   # 'x': slightly better at step 1
            log_probs[3] = -0.5   # 'y': slightly worse at step 1
        elif step == 1:
            if tokens[-1] == 2:   # after 'x'
                log_probs[EOS_TOKEN] = -5.0  # terrible step 2
            else:                 # after 'y'
                log_probs[EOS_TOKEN] = -0.1  # great step 2
        else:
            log_probs[EOS_TOKEN] = -0.1
        return log_probs

    greedy_sub = greedy_decode(suboptimal_fn, max_len=5)
    beam_sub = beam_search(suboptimal_fn, beam_width=2, max_len=5)

    # Greedy picks 'x' first (locally best); beam should find 'y' -> EOS is better overall
    greedy_total = sum(suboptimal_fn(greedy_sub[:i])[greedy_sub[i]]
                       for i in range(len(greedy_sub)))
    beam_best_score = beam_sub[0][1]
    assert beam_best_score >= greedy_total - 1e-6, (
        f"Beam search should score >= greedy: {beam_best_score:.3f} vs {greedy_total:.3f}"
    )

    print("  beam_search: all tests passed")
    print(f"    Greedy: {greedy_out} | Beam best: {beam_out[0][0]}")


if __name__ == "__main__":
    _test()
