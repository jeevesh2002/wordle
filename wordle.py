"""Information-theoretic Wordle solver toolkit.

What's here:

* Vectorised feedback (pattern) computation in NumPy. Greens are masked
  first; remaining-letter counts are decremented per position so that
  duplicate-letter answers are handled exactly the way the NYT does.
* Single-step Shannon-entropy guess selection — pick the word whose
  feedback distribution over the remaining candidates is closest to
  uniform. This maximises expected bits of information per turn.
* Optional two-step lookahead (``--lookahead 2``) that scores the top-K
  candidates by expected entropy *after* the next guess inside each
  feedback bucket.
* Hard-mode constraint (``--hard``): every guess must itself satisfy all
  prior feedback.
* Opener analysis (``--analyze-openers``): rank every word by its first-
  turn entropy against the full wordlist.
* Benchmark mode (``--bench N``) with solve rate, average guesses, and
  full turn-count distribution.

Feedback strings use ``G`` (green), ``Y`` (yellow), ``B`` (black/gray).
Internally a feedback is a base-3 int in ``[0, 243)``.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

WORD_LEN = 5
N_PATTERNS = 3 ** WORD_LEN  # 243
MAX_GUESSES = 6
DEFAULT_OPENER = "SALET"
ALL_GREEN = sum(2 * 3 ** i for i in range(WORD_LEN))  # 242
WORDS_FILE = Path(__file__).parent / "five_letter.txt"

ESC = "\033["
GREEN_BG = f"{ESC}42;30m"
YELLOW_BG = f"{ESC}43;30m"
GRAY_BG = f"{ESC}100;37m"
RESET = f"{ESC}0m"
_WEIGHTS = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


# ---------------------------------------------------------------------------
# Wordlist + encoding
# ---------------------------------------------------------------------------

def load_words(path: Path = WORDS_FILE) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        w = line.strip().upper()
        if len(w) == WORD_LEN and w.isalpha() and w not in seen:
            words.append(w)
            seen.add(w)
    if not words:
        raise SystemExit(f"No 5-letter words found in {path}")
    return words


def encode(words: Sequence[str]) -> np.ndarray:
    arr = np.empty((len(words), WORD_LEN), dtype=np.int8)
    for i, w in enumerate(words):
        for j, c in enumerate(w):
            arr[i, j] = ord(c) - 65
    return arr


def encode_word(word: str) -> np.ndarray:
    return np.array([ord(c) - 65 for c in word.upper()], dtype=np.int8)


# ---------------------------------------------------------------------------
# Vectorised feedback
# ---------------------------------------------------------------------------

def patterns_for_guess(guess_enc: np.ndarray, answers_enc: np.ndarray) -> np.ndarray:
    """Wordle feedback for one guess against many answers.

    ``guess_enc`` is shape ``(5,)`` int8. ``answers_enc`` is ``(N, 5)`` int8.
    Returns ``(N,)`` uint8 with each feedback encoded as a base-3 int.
    """
    n = answers_enc.shape[0]
    greens = answers_enc == guess_enc  # (n, 5) bool
    marks = greens.astype(np.uint8) * 2

    # Build per-answer remaining-letter counts (letters not consumed by greens).
    rem = np.zeros((n, 26), dtype=np.int16)
    for pos in range(WORD_LEN):
        not_green = ~greens[:, pos]
        idx = np.where(not_green)[0]
        if idx.size:
            np.add.at(rem, (idx, answers_enc[idx, pos]), 1)

    # Yellow pass — order matters when the guess has duplicate letters.
    for pos in range(WORD_LEN):
        not_green_here = marks[:, pos] == 0
        L = int(guess_enc[pos])
        eligible = not_green_here & (rem[:, L] > 0)
        marks[eligible, pos] = 1
        rem[eligible, L] -= 1

    return (marks.astype(np.uint16) * _WEIGHTS).sum(axis=1).astype(np.uint8)


def entropy_of_patterns(patterns: np.ndarray) -> float:
    counts = np.bincount(patterns, minlength=N_PATTERNS)
    n = counts.sum()
    if n == 0:
        return 0.0
    nz = counts[counts > 0]
    p = nz / n
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# Feedback string helpers
# ---------------------------------------------------------------------------

def fb_to_str(fb: int) -> str:
    out = []
    for _ in range(WORD_LEN):
        out.append("BYG"[fb % 3])
        fb //= 3
    return "".join(out)


def str_to_fb(s: str) -> Optional[int]:
    s = s.strip().upper()
    if len(s) != WORD_LEN or set(s) - set("GYB"):
        return None
    return sum({"B": 0, "Y": 1, "G": 2}[c] * (3 ** i) for i, c in enumerate(s))


def colorize(guess: str, fb: int) -> str:
    bgs = (GRAY_BG, YELLOW_BG, GREEN_BG)
    parts, v = [], fb
    for ch in guess:
        parts.append(f"{bgs[v % 3]} {ch} {RESET}")
        v //= 3
    return "".join(parts)


# ---------------------------------------------------------------------------
# Guess selection
# ---------------------------------------------------------------------------

def _score_pool(pool_enc: np.ndarray, candidates_enc: np.ndarray):
    g_count = pool_enc.shape[0]
    m = candidates_enc.shape[0]
    scores = np.empty(g_count, dtype=np.float64)
    patterns = np.empty((g_count, m), dtype=np.uint8)
    for i in range(g_count):
        patterns[i] = patterns_for_guess(pool_enc[i], candidates_enc)
        scores[i] = entropy_of_patterns(patterns[i])
    return scores, patterns


def best_guess(
    candidates_idx: np.ndarray,
    all_words_enc: np.ndarray,
    *,
    hard: bool = False,
    lookahead: int = 1,
    full_pool: bool = False,
    exclude_idx: Optional[np.ndarray] = None,
) -> int:
    """Pick the best next guess. Returns an index into ``all_words``."""
    m = len(candidates_idx)
    if m == 1:
        return int(candidates_idx[0])
    if m == 2:
        return int(candidates_idx[0])

    candidates_enc = all_words_enc[candidates_idx]

    if hard or not full_pool:
        pool_idx = np.asarray(candidates_idx)
    else:
        pool_idx = np.arange(all_words_enc.shape[0])
    if exclude_idx is not None and exclude_idx.size:
        pool_idx = np.setdiff1d(pool_idx, exclude_idx, assume_unique=False)
        if pool_idx.size == 0:
            raise RuntimeError("Guess pool exhausted after exclusions.")
    pool_enc = all_words_enc[pool_idx]

    scores, patterns = _score_pool(pool_enc, candidates_enc)
    cand_set = set(int(x) for x in candidates_idx)
    # Tie-break preference for candidates (a candidate could win outright).
    bonus = np.array(
        [1e-9 if int(g) in cand_set else 0.0 for g in pool_idx],
        dtype=np.float64,
    )

    if lookahead <= 1:
        return int(pool_idx[(scores + bonus).argmax()])

    # Two-step lookahead: re-rank the top-K by expected info over two turns.
    K = min(20, len(scores))
    top = np.argsort(-(scores + bonus))[:K]
    best_total = -math.inf
    best_choice = int(pool_idx[top[0]])
    for i in top:
        bucket = patterns[i]
        total = 0.0
        for pat in np.unique(bucket):
            mask = bucket == pat
            size = int(mask.sum())
            p_bucket = size / m
            if int(pat) == ALL_GREEN:
                total += p_bucket * math.log2(m)
                continue
            sub_enc = candidates_enc[mask]
            sub_scores, _ = _score_pool(sub_enc, sub_enc)
            best_next = float(sub_scores.max()) if sub_scores.size else 0.0
            total += p_bucket * (math.log2(m / size) + best_next)
        total += bonus[i]
        if total > best_total:
            best_total = total
            best_choice = int(pool_idx[i])
    return best_choice


# ---------------------------------------------------------------------------
# Game loops
# ---------------------------------------------------------------------------

def _filter_candidates(candidates_idx, all_enc, guess_enc, fb):
    cand_enc = all_enc[candidates_idx]
    pats = patterns_for_guess(guess_enc, cand_enc)
    return candidates_idx[pats == fb]


def play(answer: Optional[str], opener: str, hard: bool,
         lookahead: int, full_pool: bool) -> int:
    all_words = load_words()
    all_enc = encode(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    candidates_idx = np.arange(len(all_words))
    answer_enc = encode_word(answer)[np.newaxis] if answer else None
    excluded: list[int] = []

    for turn in range(1, MAX_GUESSES + 1):
        if turn == 1:
            guess = opener.upper()
            guess_enc = encode_word(guess)
        else:
            gi = best_guess(
                candidates_idx, all_enc,
                hard=hard, lookahead=lookahead, full_pool=full_pool,
                exclude_idx=np.array(excluded, dtype=np.int64) if excluded else None,
            )
            guess = all_words[gi]
            guess_enc = all_enc[gi]

        info = entropy_of_patterns(
            patterns_for_guess(guess_enc, all_enc[candidates_idx])
        )
        print(f"\nTurn {turn}: {len(candidates_idx)} candidate(s), "
              f"expected info {info:.2f} bits")
        print(f"  Guess: {guess}")

        if answer_enc is not None:
            fb = int(patterns_for_guess(guess_enc, answer_enc)[0])
        else:
            fb = None
            while fb is None:
                raw = input("  Feedback [GYB x5, 'skip' to drop]: ")
                if raw.strip().lower() == "skip":
                    if guess in word_to_idx:
                        excluded.append(word_to_idx[guess])
                    try:
                        gi = best_guess(
                            candidates_idx, all_enc,
                            hard=hard, lookahead=lookahead, full_pool=full_pool,
                            exclude_idx=np.array(excluded, dtype=np.int64),
                        )
                    except RuntimeError:
                        print("  No candidates left after skip.")
                        return 1
                    guess = all_words[gi]
                    guess_enc = all_enc[gi]
                    print(f"  Next guess: {guess}")
                    continue
                fb = str_to_fb(raw)
                if fb is None:
                    print("  Invalid input: enter exactly 5 chars from {G, Y, B}.")

        print(f"  Result: {colorize(guess, fb)}  ({fb_to_str(fb)})")
        if fb == ALL_GREEN:
            print(f"\nSolved in {turn} guess(es).")
            return 0
        candidates_idx = _filter_candidates(candidates_idx, all_enc, guess_enc, fb)
        if len(candidates_idx) == 0:
            print("\nNo candidates match all feedback. Inconsistent input?")
            return 1

    remaining = [all_words[i] for i in candidates_idx[:10]]
    print(f"\nFailed in {MAX_GUESSES} guesses. Remaining: {remaining}"
          f"{'...' if len(candidates_idx) > 10 else ''}")
    return 1


def bench(n: int, opener: str, hard: bool, lookahead: int,
          full_pool: bool, seed: int) -> int:
    all_words = load_words()
    all_enc = encode(all_words)
    rng = random.Random(seed)
    n = len(all_words) if n < 0 else min(n, len(all_words))
    sample = rng.sample(range(len(all_words)), n)

    opener_enc = encode_word(opener)
    distribution: Counter[int] = Counter()
    turns_taken: list[int] = []
    start = time.time()
    progress_every = max(50, n // 20)

    for done, ans_idx in enumerate(sample, 1):
        answer_enc = all_enc[ans_idx][np.newaxis]
        candidates_idx = np.arange(len(all_words))
        outcome = 0
        for turn in range(1, MAX_GUESSES + 1):
            if turn == 1:
                guess_enc = opener_enc
            else:
                gi = best_guess(
                    candidates_idx, all_enc,
                    hard=hard, lookahead=lookahead, full_pool=full_pool,
                )
                guess_enc = all_enc[gi]
            fb = int(patterns_for_guess(guess_enc, answer_enc)[0])
            if fb == ALL_GREEN:
                outcome = turn
                break
            candidates_idx = _filter_candidates(
                candidates_idx, all_enc, guess_enc, fb,
            )
            if len(candidates_idx) == 0:
                break
        distribution[outcome] += 1
        if outcome:
            turns_taken.append(outcome)
        if done % progress_every == 0 or done == n:
            avg = sum(turns_taken) / max(len(turns_taken), 1)
            print(f"  {done}/{n} games  ({time.time() - start:.1f}s, "
                  f"avg {avg:.3f})")

    elapsed = time.time() - start
    print(f"\nBenchmarked {n} games "
          f"(opener={opener.upper()}, hard={hard}, "
          f"lookahead={lookahead}, full_pool={full_pool}) in {elapsed:.1f}s")
    solved = len(turns_taken)
    print(f"Solved: {solved}/{n}  ({100 * solved / n:.2f}%)")
    if solved:
        avg = sum(turns_taken) / solved
        std = math.sqrt(sum((t - avg) ** 2 for t in turns_taken) / solved)
        print(f"Mean guesses (solved games): {avg:.3f}  (σ={std:.3f})")
    print("Distribution:")
    bar_unit = max(distribution.values()) / 40
    for k in sorted(distribution):
        label = "FAIL" if k == 0 else f"{k}"
        bar = "█" * int(distribution[k] / bar_unit) if bar_unit else ""
        print(f"  {label:>4}: {distribution[k]:5d}  {bar}")
    return 0


def analyze_openers(top_k: int) -> int:
    all_words = load_words()
    all_enc = encode(all_words)
    n = len(all_words)
    print(f"Computing first-turn entropy for {n} openers on {n} answers...")
    scores = np.empty(n, dtype=np.float64)
    start = time.time()
    for i in range(n):
        scores[i] = entropy_of_patterns(patterns_for_guess(all_enc[i], all_enc))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}  ({time.time() - start:.1f}s)")
    print(f"Done in {time.time() - start:.1f}s\n")
    order = np.argsort(-scores)
    print(f"Top {top_k} openers by single-step entropy:")
    print(f"  {'rank':<6}{'word':<10}{'bits':>8}")
    for rank, i in enumerate(order[:top_k], 1):
        print(f"  {rank:<6}{all_words[int(i)]:<10}{scores[int(i)]:>8.3f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Information-theoretic Wordle solver."
    )
    p.add_argument("--answer", help="Self-play against this known answer (no prompts).")
    p.add_argument("--opener", default=DEFAULT_OPENER,
                   help=f"First guess (default: {DEFAULT_OPENER}).")
    p.add_argument("--hard", action="store_true",
                   help="Hard mode: every guess must itself satisfy all prior feedback.")
    p.add_argument("--lookahead", type=int, default=1, choices=[1, 2],
                   help="Search depth for guess selection (default: 1).")
    p.add_argument("--full-pool", action="store_true",
                   help="Search the entire wordlist as guesses (slower, marginal gain).")
    p.add_argument("--bench", type=int, metavar="N",
                   help="Benchmark over N random answers. Pass -1 for all words.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for --bench sampling (default: 42).")
    p.add_argument("--analyze-openers", action="store_true",
                   help="Rank every word by first-turn entropy and print the top --top.")
    p.add_argument("--top", type=int, default=15,
                   help="Top-K to show for --analyze-openers (default: 15).")
    args = p.parse_args()

    if args.analyze_openers:
        return analyze_openers(args.top)
    if args.bench is not None:
        return bench(args.bench, args.opener, args.hard, args.lookahead,
                     args.full_pool, args.seed)
    return play(args.answer, args.opener, args.hard, args.lookahead, args.full_pool)


if __name__ == "__main__":
    sys.exit(main())
