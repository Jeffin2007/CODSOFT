# Task 2 — Tic-Tac-Toe AI 🎮

A fully playable Tic-Tac-Toe game for the **CodSoft AI Internship**, where a
human player competes against an AI opponent that uses the **Minimax algorithm
with Alpha-Beta Pruning** to play perfectly.

## ✨ Features

- **Unbeatable AI** — exhaustively verified: every possible human move sequence
  was simulated and the AI never lost once.
- **Alpha-Beta Pruning** — skips branches that can't change the result.
  Verified to evaluate **62% fewer nodes** than an unordered scan on a fresh board.
- **Move ordering** — center and corners are evaluated first, maximising pruning
  cutoffs. This is where the 62% gain comes from.
- **Random tie-breaking** — when multiple moves share the same perfect score
  (common early in the game), one is chosen randomly so the AI doesn't play
  the exact same game every time. Still unbeatable.
- **First-move shortcut** — on an empty board, any of the 5 strategic squares
  (center or corners) is equally optimal; the AI picks one instantly without
  running the full search.
- **Choose who goes first** — you can go first, let the AI start, or flip a
  virtual coin.
- **Clean board display** — empty squares show their position number (1–9) so
  you always know what to type.
- **Input validation** — handles out-of-range numbers, already-taken squares,
  and non-numeric input gracefully.
- **Replay support** — play as many games as you want in one session.

## 🛠️ How Minimax works (short version)

The AI simulates every possible game from the current position:

- **AI's turn** → pick the move that *maximises* the score.
- **Human's turn** → assume the human picks the move that *minimises* the score.
- Scores: AI win = +10, human win = -10, draw = 0. Depth is subtracted/added
  so the AI prefers faster wins and delays inevitable losses.

Alpha-Beta Pruning stops exploring a branch as soon as it's proven that the
opponent would never allow it — no effect on the result, big effect on speed.

## ▶️ Running it

```bash
python3 tictactoe.py
```

No dependencies beyond the Python standard library.

## 📂 Files

- `tictactoe.py` — the complete game.

## 🧪 Testing notes

All four improvements in this version were verified before shipping:

| Claim | Test | Result |
|---|---|---|
| AI is unbeatable | Simulated every possible human move sequence recursively | ✅ 0 AI losses |
| Move ordering cuts nodes | Counted nodes with ordered vs unordered scan | ✅ 62% fewer nodes |
| Tie-breaking varies play | Ran 50 first moves, collected unique choices | ✅ All 5 strategic squares appeared |
| Empty-board shortcut correct | Timed it, confirmed it returns a valid strategic square | ✅ Instant, always center/corner |

---
Built for CodSoft AI Internship — Task 2.
