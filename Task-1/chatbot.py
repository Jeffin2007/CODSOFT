"""
CODSOFT AI Internship - Task 1
Rule-Based Chatbot (Production-Grade Framework)
"""

import re
import random
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

INTENTS: Dict[str, Dict[str, Any]] = {
    "greeting": {
        "patterns": [
            r"\b(hi|hello|hey|yo|sup|hiya|howdy)\b",
            r"\bgood\s*(morning|afternoon|evening)\b",
        ],
        "responses": [
            "Hey there! How can I help you today?",
            "Hello! What's on your mind?",
            "Hi! Great to see you here.",
            "Hey! What can I do for you?",
        ],
    },
    "goodbye": {
        "patterns": [r"\b(bye|goodbye|see\s*ya|see\s*you|farewell|catch\s*you\s*later)\b"],
        "responses": ["Goodbye! Have a great day ahead.", "See you later! Take care.", "Bye! Come back anytime you have questions."],
    },
    "thanks": {
        "patterns": [r"\b(thanks|thank\s*you|thx|appreciate\s*it|ty)\b"],
        "responses": ["You're welcome!", "Anytime, happy to help!", "No problem at all!"],
    },
    "how_are_you": {
        "patterns": [
            r"\bhow\s*(are|r)\s*(you|u)\b",
            r"\bhow('s| is)\s*it\s*going\b",
            r"\bhow\s*('?s| is| are) (everything|life|things)\b",
        ],
        "responses": [
            "I'm doing great, thanks for asking! How about you?",
            "All good on my end! How are you doing?",
            "I'm just a bunch of rules, but I'm running smoothly! And you?",
        ],
        "set_context": "awaiting_wellbeing"
    },
    "user_wellbeing_reply": {
        "patterns": [
            r"\b(i'?m|i\s+am)\s+(good|fine|great|ok|okay|alright|not\s*great|tired|sad|happy)\b",
        ],
        "required_context": "awaiting_wellbeing",
        "context_patterns": [
            r"^\s*(good|fine|great|ok|okay|alright|not\s*great|tired|sad|happy)\s*$",
        ],
        "responses": [
            "Glad to hear that! Let me know if there's anything you need.",
            "Good to know. What would you like help with today?",
            "Thanks for sharing! What can I do for you next?"
        ],
    },
    "name_query": {
        "patterns": [
            r"\bwhat(\'?s| is) your name\b",
            r"\bwho are you\b",
            r"\bwhat are you called\b",
        ],
        "responses": [
            "I'm a rule-based chatbot built for the CodSoft AI Internship!",
            "You can call me CodBot — I run on pattern matching, not magic.",
        ],
    },
    "tell_my_name": {
        "patterns": [
            r"\bmy name is\s+([a-zA-Z0-9]+)\b",
            r"\bi\s+am\s+([a-zA-Z0-9]+)\b",
            r"\bi'?m\s+([a-zA-Z0-9]+)\b",
            r"\bcall me\s+([a-zA-Z0-9]+)\b",
        ],
        "responses": [
            "Nice to meet you, {name}!",
            "Got it, I'll remember you as {name}.",
            "Welcome, {name}! How can I help you today?",
        ],
    },
    "recall_my_name": {
        "patterns": [
            r"\bwhat(\'?s| is) my name\b",
            r"\bdo you remember my name\b",
            r"\bwho am i\b",
        ],
        "responses": ["You told me your name is {name}!", "You're {name}, of course!"],
    },
    "time_query": {
        "patterns": [r"\bwhat(\'?s| is) the time\b", r"\bcurrent time\b", r"\bwhat time is it\b"],
        "responses": ["The current time is {time}."],
    },
    "date_query": {
        "patterns": [r"\bwhat(\'?s| is) the date\b", r"\btoday'?s date\b", r"\bwhat day is it\b"],
        "responses": ["Today's date is {date}."],
    },
    "bot_capability": {
        "patterns": [
            r"\bwhat can you do\b", r"\bwhat do you do\b", r"\bhelp me\b", r"^help$", r"\bwhat are your features\b",
        ],
        "responses": [
            "I can chat with you, tell you the time/date, remember your name, "
            "and answer a few common questions. Try asking me 'what's your name?' "
            "or just say hi!",
            "I'm a simple rule-based assistant — I recognize greetings, farewells, "
            "questions about myself, time/date queries, and more. Type 'help' anytime!",
        ],
    },
    "ai_query": {
        "patterns": [r"\bwhat is ai\b", r"\bwhat is artificial intelligence\b", r"\bdefine ai\b"],
        "responses": [
            "Artificial Intelligence is the simulation of human intelligence "
            "by machines, especially computer systems, so they can perform tasks "
            "like reasoning, learning, and decision-making.",
        ],
    },
    "bot_feelings": {
        "patterns": [
            r"\bdo you have feelings\b", r"\bare you (alive|real|human|conscious)\b", r"\bare you a robot\b",
        ],
        "responses": [
            "I don't have feelings — I'm just matching patterns and picking "
            "responses from a list! But I appreciate you asking.",
            "Nope, no feelings here. I'm purely rule-based logic.",
        ],
    },
    "compliment": {
        "patterns": [
            r"\b(you'?re|youre|you are) (great|awesome|cool|smart|amazing|the best)\b",
            r"\bgood (job|bot)\b",
        ],
        "responses": ["Aw, thank you! I do my best with the rules I was given.", "That's kind of you to say!"],
    },
    "insult": {
        "patterns": [r"\b(you'?re|youre|you are) (dumb|stupid|useless|bad|terrible)\b"],
        "responses": [
            "Ouch! I'll try to do better. Maybe rephrase and I'll give it another shot?",
            "Sorry to disappoint — I'm pretty limited since I just follow rules!",
        ],
    },
    "joke_request": {
        "patterns": [r"\btell me a joke\b", r"\bmake me laugh\b", r"\bsay something funny\b"],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "I'd tell you a UDP joke, but you might not get it.",
            "Why did the AI cross the road? It was following a gradient descent.",
        ],
    },
}

for intent, data in INTENTS.items():
    data["compiled_patterns"] = [re.compile(p, re.IGNORECASE) for p in data.get("patterns", [])]
    data["compiled_context_patterns"] = [
        re.compile(p, re.IGNORECASE) for p in data.get("context_patterns", [])
    ]

EXIT_PATTERNS = [re.compile(r"^\s*(exit|quit|stop|end)\s*$", re.IGNORECASE)]

FALLBACK_RESPONSES = [
    "I'm not quite sure I understand. Could you rephrase that?",
    "Hmm, I don't have a rule for that yet. Try asking something else!",
    "Sorry, that one's outside my rule set. Type 'help' to see what I can do.",
    "I didn't catch that. Could you try saying it differently?",
]

NON_NAME_WORDS = {
    "good", "fine", "great", "ok", "okay", "alright", "tired", "sad", "happy",
    "bored", "busy", "sorry", "done", "back", "here", "ready", "confused",
    "not", "still", "just", "also", "really", "very", "kind", "sure", "a", "an"
}

def clean_input(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_intent(cleaned_text: str, current_context: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    scores: Dict[str, int] = {}
    captured_groups: Dict[str, str] = {}

    for intent_name, intent_data in INTENTS.items():
        req_context = intent_data.get("required_context")
        if req_context and current_context == req_context:
            for pattern in intent_data["compiled_context_patterns"]:
                if pattern.search(cleaned_text):
                    return intent_name, None

        for pattern in intent_data["compiled_patterns"]:
            match = pattern.search(cleaned_text)
            if match:
                if intent_name == "tell_my_name" and match.groups():
                    captured_val = match.group(1).lower()
                    if captured_val in NON_NAME_WORDS:
                        continue
                weight = 3 if req_context and current_context == req_context else 1
                scores[intent_name] = scores.get(intent_name, 0) + weight
                if match.groups():
                    captured_groups[intent_name] = match.group(1)

    if not scores:
        return None, None

    best_intent = max(scores, key=scores.get)
    return best_intent, captured_groups.get(best_intent)

def generate_response(intent_name: str, captured_value: Optional[str], memory: Dict[str, Any]) -> str:
    response_template = random.choice(INTENTS[intent_name]["responses"])

    if intent_name == "tell_my_name":
        # Every tell_my_name pattern requires [a-zA-Z0-9]+ (one or more chars),
        # so captured_value should always be present in normal operation. This
        # check is a defensive guard: if a future edit ever makes the capture
        # group optional, we ask the user to repeat themselves instead of
        # falsely claiming we saved a name we don't actually have.
        if captured_value:
            memory["name"] = captured_value.capitalize()
            return response_template.format(name=memory["name"])
        return "Sorry, I didn't quite catch your name — could you repeat that?"

    if intent_name == "recall_my_name":
        name = memory.get("name")
        if name:
            return response_template.format(name=name)
        return "I don't think you've told me your name yet! What should I call you?"

    if intent_name == "time_query":
        return response_template.format(time=datetime.now().strftime("%I:%M %p"))

    if intent_name == "date_query":
        return response_template.format(date=datetime.now().strftime("%B %d, %Y"))

    return response_template

def run_chatbot():
    memory: Dict[str, Any] = {"context": None}
    print("=" * 60)
    print(" CodBot — Production-Ready Rule Chatbot")
    print("=" * 60)
    print("Type 'help' to see what I can do, or 'exit' to end the chat.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCodBot: Session interrupted. Goodbye!")
            break

        if not user_input:
            print("CodBot: Say something — I'm listening!")
            continue

        cleaned = clean_input(user_input)

        if any(p.search(cleaned) for p in EXIT_PATTERNS):
            print("CodBot: Goodbye! Thanks for chatting.")
            break

        intent_name, captured_value = detect_intent(cleaned, memory["context"])

        if intent_name is None:
            print(f"CodBot: {random.choice(FALLBACK_RESPONSES)}")
            memory["context"] = None
            continue

        reply = generate_response(intent_name, captured_value, memory)
        print(f"CodBot: {reply}")
        memory["context"] = INTENTS[intent_name].get("set_context")

if __name__ == "__main__":
    run_chatbot()
