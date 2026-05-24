# Wordle Solver

Information-theoretic Wordle solver. Each guess is chosen to maximise the
expected information gain (Shannon entropy) over the remaining candidate
words, so the search space shrinks as fast as possible.

Benchmarked on the bundled 12,891-word dictionary (much larger than the
~2,300-word NYT answer list), opening with `SALET`:

- Solve rate: ~98%
- Average guesses on solved games: ~4.2

## Usage

Interactive mode — the solver suggests a guess, you play it on Wordle and
type back the colour feedback:

```
python3 wordle.py
```

Feedback is 5 characters using:
- `G` — green (right letter, right spot)
- `Y` — yellow (right letter, wrong spot)
- `B` — black/gray (letter not in word)

If Wordle rejects a suggestion as not in its dictionary, type `skip` to
drop that word and get the next-best guess.

Self-play against a known answer (no prompts):

```
python3 wordle.py --answer NIECE
```

Use a different opener:

```
python3 wordle.py --opener CRANE
```

Benchmark across N random answers:

```
python3 wordle.py --bench 200
```
