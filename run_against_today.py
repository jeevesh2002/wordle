"""Drive wordle.py against today's Wordle answer non-interactively."""
import os
import pty
import re
import select
import shutil
import subprocess
import sys

ANSWER = "NIECE"
SOURCE = "five_letter.txt"
BACKUP = "five_letter.txt.bak"
ACCEPT_PROMPT = "Enter YES if word is accepted else NO :"
COLOR_PROMPT = "Enter Colors:"
WORD_RE = re.compile(r"^[A-Z]{5}$")


def compute_colors(guess: str, answer: str) -> str:
    guess = guess.upper()
    answer = answer.upper()
    colors = ["B"] * 5
    remaining = list(answer)
    for i in range(5):
        if guess[i] == remaining[i]:
            colors[i] = "G"
            remaining[i] = None
    for i in range(5):
        if colors[i] == "B" and guess[i] in remaining:
            colors[i] = "Y"
            remaining[remaining.index(guess[i])] = None
    return "".join(colors)


def read_until(fd: int, marker: str, timeout: float = 30.0) -> str:
    buf = ""
    while marker not in buf:
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            raise TimeoutError(f"Timed out waiting for {marker!r}\nBuffered:\n{buf}")
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk.decode(errors="replace")
    return buf


def last_word(text: str) -> str:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if WORD_RE.match(line):
            return line
    raise ValueError(f"No 5-letter word found in:\n{text}")


def main() -> int:
    shutil.copy(SOURCE, BACKUP)
    transcript_lines = []
    guesses = []
    solved = False
    try:
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            [sys.executable, "-u", "wordle.py"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)

        for round_no in range(1, 7):
            chunk = read_until(master_fd, ACCEPT_PROMPT)
            transcript_lines.append(chunk)
            word = last_word(chunk[: chunk.index(ACCEPT_PROMPT)])
            os.write(master_fd, b"YES\n")
            transcript_lines.append("> YES\n")

            chunk = read_until(master_fd, COLOR_PROMPT)
            transcript_lines.append(chunk)
            colors = compute_colors(word, ANSWER)
            os.write(master_fd, (colors + "\n").encode())
            transcript_lines.append(f"> {colors}\n")
            guesses.append((word, colors))
            if colors == "GGGGG":
                solved = True
                break

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=2)
    finally:
        shutil.move(BACKUP, SOURCE)

    print("=== Transcript ===")
    print("".join(transcript_lines))
    print("\n=== Summary ===")
    print(f"Answer: {ANSWER}")
    print(f"Attempts: {len(guesses)}")
    for i, (g, c) in enumerate(guesses, 1):
        print(f"  {i}. {g}  ->  {c}")
    print(f"Result: {'SOLVED' if solved else 'FAILED in 6 guesses'}")
    return 0 if solved else 1


if __name__ == "__main__":
    sys.exit(main())
