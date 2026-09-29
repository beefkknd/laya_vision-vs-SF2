"""System 2's model: Qwen 3.8 27B (Q4, MTP) served by omlx on this machine, OpenAI-style chat API. Every request and
reply (including Qwen's reasoning) is saved under logs/system2/ so any lesson can be traced to the prompt behind it."""
import json
import os
import re
import time
import urllib.request
from typing import Dict, List

from ..config import QWEN_MAX_TOKENS, QWEN_MODEL, QWEN_TEMPERATURE, QWEN_THINKING, QWEN_URL

LOG_DIR = os.path.join("logs", "system2")


def chat(messages: List[Dict], task: str, max_tokens: int = QWEN_MAX_TOKENS, temperature: float = QWEN_TEMPERATURE,
         thinking: bool = QWEN_THINKING, timeout: float = 300.0) -> str:
    """Qwen's reply text. Raises on an HTTP error, a timeout, or an empty or cut-off reply."""
    body = {"model": QWEN_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(QWEN_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    choice = out["choices"][0]
    content = choice["message"].get("content") or ""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "%s_%s.json" % (time.strftime("%Y%m%d-%H%M%S"), task))
    with open(path, "w") as f:
        json.dump({"task": task, "seconds": round(time.time() - t, 1), "request": body, "reply": content,
                   "reasoning": choice["message"].get("reasoning_content"), "finish": choice.get("finish_reason"),
                   "usage": out.get("usage")}, f, indent=1)
    if choice.get("finish_reason") != "stop" or not content.strip():
        raise RuntimeError("Qwen reply for %s was %s / empty (see %s)" % (task, choice.get("finish_reason"), path))
    return content


def json_reply(text: str) -> Dict:
    """The JSON object in a reply (a ```json fence or the outermost {...})."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fence.group(1) if fence else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)
