# Task 1 — Rule-Based Chatbot 🤖

A rule-based chatbot built for the **CodSoft AI Internship**. It uses regex pattern
matching with a scored intent-detection system (rather than a plain if-else chain)
to understand user input and reply naturally.

## ✨ Features

* **Pattern matching, not keyword-only matching** — recognizes phrasing variations
(`hi`, `hello`, `hey`, `good morning` all map to the same greeting intent).
* **Scored intent detection** — if input matches multiple rules, the bot picks the
most relevant one instead of just the first match it finds.
* **Pre-compiled regex** — all patterns are compiled once at startup instead of
on every message, for faster matching as the rule set grows.
* **Data-driven context engine** — conversation state ("the bot just asked X, so
interpret a vague reply as answering X") is configured per-intent via
`required\_context` / `context\_patterns` / `set\_context` keys, not hardcoded
`if` checks in the matching logic. Verified by adding a second, unrelated
context-gated intent at runtime and confirming the two contexts don't leak
into each other — no changes to `detect\_intent` were needed.
* **Session memory** — remembers your name once you introduce yourself and can
recall it later in the conversation.
* **Dynamic responses** — time/date queries, varied response pools so the bot
doesn't repeat the same line every time.
* **Graceful fallback** — unrecognized input gets a friendly "I didn't catch that"
reply instead of crashing or going silent.
* **Easy to extend** — every intent is a single entry in the `INTENTS` dictionary;
add a new one without touching any other code.

## 🛠️ How it works

1. **Clean input** — lowercase, strip punctuation (keep apostrophes for contractions).
2. **Context check** — if the bot is waiting on a specific kind of reply (e.g. it
just asked "how about you?"), check for that narrow case first.
3. **Detect intent** — check the cleaned text against every intent's regex patterns,
scoring each match.
4. **Generate response** — pick the highest-scoring intent, fill in any dynamic
placeholders (name, time, date), and return a randomly chosen response from
that intent's pool.

## ▶️ Running it

```bash
python3 chatbot.py
```

Type naturally — try `hi`, `what can you do`, `my name is <yourname>`,
`what's my name`, `tell me a joke`, or `what is AI`. Type `exit` to quit.

## 📂 Files

* `chatbot.py` — the complete chatbot implementation.

## 🧪 Testing notes

This version was stress-tested against \~50 phrasing variants across every intent,
including ambiguous edge cases (e.g. distinguishing "my name is Arjun" from "I'm
tired", and "good job" from a standalone wellbeing reply like "good"). Two bugs
were found and fixed during testing:

1. An early draft of the wellbeing-reply rule matched the bare words
"good"/"fine"/"great" anywhere in a message, which caused phrases like
"good job" to be misread as a wellbeing reply instead of a compliment.
Fixed by gating that pattern to only activate in the specific conversational
context it was designed for.
2. A later "safety guard" intended to handle a missing name-capture silently
claimed "I've logged your name" even when nothing was actually saved to
memory. In practice this branch is unreachable today (every name pattern
requires at least one character), but it's a real trap for future edits —
fixed to honestly ask the user to repeat themselves instead of lying about
what was stored.



Built for CodSoft AI Internship — Task 1.

