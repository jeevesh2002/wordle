"""Information-theoretic Wordle solver.

Picks each guess to maximise expected information gain (entropy) over the
remaining candidate words. Interactive by default; can also self-play
against a known answer or benchmark itself across a random sample.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

WORD_LEN = 5
MAX_GUESSES = 6
DEFAULT_OPENER = "SALET"
WORDS_FILE = Path(__file__).parent / "five_letter.txt"

ESC = "\033["
GREEN_BG = f"{ESC}42;30m"
YELLOW_BG = f"{ESC}43;30m"
GRAY_BG = f"{ESC}100;37m"
RESET = f"{ESC}0m"


def load_words(path: Path) -> list[str]:
    words: list[str] = []
    for line in path.read_text().splitlines():
        w = line.strip().upper()
        if len(w) == WORD_LEN and w.isalpha():
            words.append(w)
    if not words:
        raise SystemExit(f"No 5-letter words found in {path}")
    return words


def feedback(guess: str, answer: str) -> int:
    """Wordle feedback encoded as a base-3 int (digit i: 0=gray, 1=yellow, 2=green)."""
    marks = [0, 0, 0, 0, 0]
    rem = [answer[0], answer[1], answer[2], answer[3], answer[4]]
    for i in range(WORD_LEN):
        if guess[i] == rem[i]:
            marks[i] = 2
            rem[i] = ""
    for i in range(WORD_LEN):
        if marks[i] == 0:
            g = guess[i]
            for j in range(WORD_LEN):
                if rem[j] == g:
                    marks[i] = 1
                    rem[j] = ""
                    break
    return marks[0] + 3 * marks[1] + 9 * marks[2] + 27 * marks[3] + 81 * marks[4]


def fb_to_str(fb: int) -> str:
    out = []
    for _ in range(WORD_LEN):
        d = fb % 3
        fb //= 3
        out.append("BYG"[d])
    return "".join(out)


def str_to_fb(s: str) -> int | None:
    s = s.strip().upper()
    if len(s) != WORD_LEN or set(s) - {"G", "Y", "B"}:
        return None
    table = {"B": 0, "Y": 1, "G": 2}
    return sum(table[c] * (3 ** i) for i, c in enumerate(s))


def colorize(guess: str, fb: int) -> str:
    bgs = (GRAY_BG, YELLOW_BG, GREEN_BG)
    parts = []
    v = fb
    for ch in guess:
        parts.append(f"{bgs[v % 3]} {ch} {RESET}")
        v //= 3
    return "".join(parts)


def filter_candidates(words: list[str], guess: str, fb: int) -> list[str]:
    return [w for w in words if feedback(guess, w) == fb]


def expected_info(guess: str, candidates: list[str]) -> float:
    buckets: Counter[int] = Counter()
    for w in candidates:
        buckets[feedback(guess, w)] += 1
    n = len(candidates)
    return -sum((c / n) * math.log2(c / n) for c in buckets.values())


def best_guess(candidates: list[str], all_words: list[str]) -> str:
    if len(candidates) <= 2:
        return candidates[0]
    cand_set = set(candidates)
    # Restrict the search pool to candidates for speed. With a strong opener
    # this is usually sufficient to solve in <=5 turns and avoids the
    # 12k * N feedback evaluations that searching all_words would require.
    pool = candidates
    return max(pool, key=lambda g: (expected_info(g, candidates), g in cand_set))


def play(answer: str | None, opener: str | None) -> int:
    all_words = load_words(WORDS_FILE)
    candidates = list(all_words)

    for turn in range(1, MAX_GUESSES + 1):
        if turn == 1 and opener:
            guess = opener.upper()
        else:
            guess = best_guess(candidates, all_words)

        ei = expected_info(guess, candidates) if candidates else 0.0
        print(f"\nTurn {turn}: {len(candidates)} candidate(s), "
              f"expected info {ei:.2f} bits")
        print(f"  Guess: {guess}")

        if answer:
            fb = feedback(guess, answer.upper())
        else:
            fb = None
            while fb is None:
                raw = input("  Feedback [GYB x5, or 'skip' to drop this word]: ")
                if raw.strip().lower() == "skip":
                    candidates = [w for w in candidates if w != guess]
                    if not candidates:
                        print("  No candidates left after skip.")
                        return 1
                    guess = best_guess(candidates, all_words)
                    print(f"  Next guess: {guess}")
                    continue
                fb = str_to_fb(raw)
                if fb is None:
                    print("  Invalid input: enter exactly 5 chars from {G, Y, B}.")

        print(f"  Result: {colorize(guess, fb)}  ({fb_to_str(fb)})")
        if fb_to_str(fb) == "G" * WORD_LEN:
            print(f"\nSolved in {turn} guess(es).")
            return 0

        candidates = filter_candidates(candidates, guess, fb)
        if not candidates:
            print("\nNo candidates match all feedback. Inconsistent input?")
            return 1

    print(f"\nFailed in {MAX_GUESSES} guesses. "
          f"Last candidates: {candidates[:10]}{'...' if len(candidates) > 10 else ''}")
    return 1


def bench(n: int, opener: str, seed: int) -> int:
    all_words = load_words(WORDS_FILE)
    rng = random.Random(seed)
    sample = rng.sample(all_words, min(n, len(all_words)))
    distribution: Counter[int] = Counter()
    solved_turns: list[int] = []
    start = time.time()

    for answer in sample:
        candidates = list(all_words)
        outcome = 0
        for turn in range(1, MAX_GUESSES + 1):
            if turn == 1:
                guess = opener.upper()
            else:
                guess = best_guess(candidates, all_words)
            fb = feedback(guess, answer)
            if fb_to_str(fb) == "G" * WORD_LEN:
                outcome = turn
                break
            candidates = filter_candidates(candidates, guess, fb)
            if not candidates:
                break
        distribution[outcome] += 1
        if outcome:
            solved_turns.append(outcome)

    elapsed = time.time() - start
    print(f"Benchmarked {len(sample)} games (opener={opener.upper()}, "
          f"seed={seed}) in {elapsed:.1f}s")
    print(f"Solved: {len(solved_turns)}/{len(sample)} "
          f"({100 * len(solved_turns) / len(sample):.1f}%)")
    if solved_turns:
        avg = sum(solved_turns) / len(solved_turns)
        print(f"Average guesses (solved games): {avg:.2f}")
    print("Distribution:")
    for k in sorted(distribution):
        label = f"{k}" if k else "FAIL"
        print(f"  {label}: {distribution[k]}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Information-theoretic Wordle solver."
    )
    p.add_argument("--answer", help="Self-play against this known answer (no prompts).")
    p.add_argument("--opener", default=DEFAULT_OPENER,
                   help=f"First guess (default: {DEFAULT_OPENER}).")
    p.add_argument("--bench", type=int, metavar="N",
                   help="Benchmark over N random answers and print stats.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for --bench (default: 42).")
    args = p.parse_args()

    if args.bench is not None:
        return bench(args.bench, args.opener, args.seed)
    return play(args.answer, args.opener)


if __name__ == "__main__":
    sys.exit(main())
