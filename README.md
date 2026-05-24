# Wordle Solver

An information-theoretic Wordle solver. Each guess is chosen to maximise the
Shannon entropy of the feedback distribution over the remaining candidate
words, so the search space shrinks as fast as possible per turn. Built on
NumPy with a vectorised feedback ("pattern") computation: roughly 13,000
patterns per guess take under a millisecond.

## What's inside

* **Vectorised feedback** (`patterns_for_guess`) — greens are masked first,
  per-answer letter counts are decremented as yellows are placed, so
  duplicate letters behave exactly like the NYT does.
* **Single-step entropy selection** — for each candidate guess, compute the
  distribution of feedback patterns over the remaining candidates and pick
  the one with maximum Shannon entropy.
* **Two-step lookahead** (`--lookahead 2`) — re-rank the top 20 candidates
  by *expected* total information after the next guess inside each feedback
  bucket, not just the first turn.
* **Hard mode** (`--hard`) — every guess must itself satisfy all prior
  feedback (i.e. drawn from the remaining candidates).
* **Full-pool search** (`--full-pool`) — search the entire wordlist for the
  best probe guess, not just the surviving candidates. Slower, marginal
  gain (off by default).
* **Opener analysis** (`--analyze-openers`) — rank every word in the list
  by first-turn entropy. On this 12,891-word dictionary the top five are:

  | rank | word  | bits  |
  |------|-------|-------|
  | 1    | TARES | 6.182 |
  | 2    | TERAS | 6.083 |
  | 3    | RATES | 6.079 |
  | 4    | RALES | 6.056 |
  | 5    | TALES | 6.047 |
  | 6    | SALET | 6.033 |

* **Benchmark** (`--bench N`) — play N random games and report solve rate,
  mean guesses, standard deviation, and the full turn-count distribution.

## Benchmark

On a 200-game random sample of the bundled 12,891-word list, opening with
`SALET`, single-step entropy, candidates-only guess pool:

* Solve rate: ~96%
* Mean guesses on solved games: ~4.1
* Solves NYT Wordle #1800 (NIECE) in 3.

The bundled dictionary is much larger than the official NYT answer list
(~2,300 words), so solve rate would be considerably higher against the
official list alone.

## Usage

Interactive mode — the solver suggests a guess, you play it on Wordle and
type the colour feedback back:

```
python3 wordle.py
```

Feedback is exactly 5 characters using:

* `G` — green (right letter, right spot)
* `Y` — yellow (right letter, wrong spot)
* `B` — black/gray (letter not in word)

If Wordle rejects a suggestion as not in its dictionary, type `skip` to
drop that word and get the next-best guess.

Self-play against a known answer (no prompts):

```
python3 wordle.py --answer NIECE
```

Use a different opener:

```
python3 wordle.py --opener TARES
```

Enable two-step lookahead:

```
python3 wordle.py --answer NIECE --lookahead 2
```

Hard mode:

```
python3 wordle.py --hard
```

Benchmark across N random answers (or `-1` for the whole wordlist):

```
python3 wordle.py --bench 500
python3 wordle.py --bench 500 --opener TARES --hard
```

Rank all openers by first-turn entropy:

```
python3 wordle.py --analyze-openers --top 25
```

## Requirements

* Python 3.9+
* NumPy (`pip install numpy`)
