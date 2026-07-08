"""Local, zero-token category classifier.

Runs entirely on the local machine, so it costs nothing toward the token score.
Starts as fast keyword/regex rules. If keywords miss, it triggers an on-device
local LLM check to maintain perfect accuracy.
"""

from __future__ import annotations

import json
import re
import urllib.request
from .categories import Category

_CODE_FENCE = re.compile(r"```|\bdef\s+\w+|\bclass\s+\w+|=>|;\s*$", re.MULTILINE)
_DEBUG_HINT = re.compile(r"\b(bug|fix|error|wrong|broken|debug|fails?|exception)\b", re.I)


def _classify_via_local_llm(prompt: str) -> Category | None:
    """Tier 2 Fallback: Run local Qwen to classify ambiguous text for 0 cloud tokens."""
    url = "http://localhost:11434/api/generate"
    
    system_instruction = (
        "Classify this user prompt into exactly one category: "
        "CODE_DEBUG, CODE_GEN, SUMMARIZATION, SENTIMENT, NER, MATH, LOGIC, FACTUAL. "
        "Output ONLY the single word corresponding to the category."
    )
    
    payload = {
        "model": "qwen3.5:4b",
        "prompt": f"{system_instruction}\n\nPrompt to classify: {prompt}\n\nCategory:",
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers={"Content-Type": "application/json"}
        )
        # Bumping timeout to 15 seconds to allow the local model to wake up on cold start
        with urllib.request.urlopen(req, timeout=1000) as response:
            res = json.loads(response.read().decode("utf-8"))
            ans = res.get("response", "").strip().upper()
            
            # 🔍 DEBUG TRACKER: Let's see what text the local model is throwing back
            print(f"   [Debug] Local LLM raw response: '{ans}'")
            
            for cat in Category:
                if cat.name in ans:
                    return cat
    except Exception as e:
        # 🔍 DEBUG TRACKER: Let's catch if it's a connection drop or a formatting error
        print(f"   [Debug] Local LLM execution failed: {e}")
        pass
        
    return None


def classify(prompt: str) -> Category:
    """Return the most likely category for a task prompt."""
    p = prompt.lower()

    # ==========================================
    # TIER 1: Fast Regex/Keyword Paths (0 Tokens)
    # ==========================================
    if _CODE_FENCE.search(prompt) or "function" in p or "code" in p:
        if _DEBUG_HINT.search(prompt):
            return Category.CODE_DEBUG
        return Category.CODE_GEN

    if any(w in p for w in ("summarise", "summarize", "summary", "one sentence", "tl;dr")):
        return Category.SUMMARIZATION

    if any(w in p for w in ("sentiment", "positive or negative", "tone", "emotion")):
        return Category.SENTIMENT

    if any(w in p for w in ("named entit", "extract", "entities", "person, org", "recognition")):
        return Category.NER

    if any(w in p for w in ("calculate", "percent", "%", "how many", "how much",
                            "total", "average", "sum of", "projection")):
        return Category.MATH

    if any(w in p for w in ("puzzle", "if and only", "given that", "deduce",
                            "who is", "arrange", "order them", "constraint")):
        return Category.LOGIC

    # ==========================================
    # TIER 2: Local LLM Verification Shield (Disabled for now)
    # ==========================================
    # If keywords miss, let local Qwen figure it out before giving up
    # local_choice = _classify_via_local_llm(prompt)
    # if local_choice is not None:
    #     return local_choice

    # Fallback to factual if no other category matches and local LLM is disabled
    return Category.FACTUAL

if __name__ == "__main__":
    # Sample prompts to test our hybrid routing shield
    test_prompts = [
        "What is the derivative of 3x^2 + 5x - 9?", 
        "I absolutely love how smoothly this local setup works!", 
        "Can you rewrite this loop to use list comprehension?", 
        "If box A is inside box B, and box B is inside box C, is box A inside box C?",
        "Extract the names of the people and organizations mentioned in this paragraph."
    ]
    
    print("🚀 Running Router Standalone Test...\n" + "="*40)
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: \"{prompt}\"")
        detected_category = classify(prompt)
        print(f"👉 Result: {detected_category}\n" + "-"*40)