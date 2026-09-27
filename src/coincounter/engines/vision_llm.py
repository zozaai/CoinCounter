"""Zero-shot counting by asking Gemini on the free tier of the Gemini API; the count is parsed from the reply."""
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

from PIL import Image

from ..exceptions import InferenceError, UnavailableEngineError
from ..types import CountResult
from .base import BaseEngine

# Gemini's OpenAI-compatible endpoint keeps the request format provider-neutral.
API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_PROMPT = ("Count the coins in this photo. Count every coin exactly once, including partly hidden "
                  'or overlapping coins. Reply with JSON only, in the form {"count": <integer>}.')
RETRY_STATUSES = {429, 500, 502, 503, 504}


def parse_count(text):
    """Return the count from a JSON reply, falling back to the only integer in the text."""
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            value = json.loads(match.group(0)).get("count")
        except (ValueError, AttributeError):
            value = None
        if type(value) is int and value >= 0:
            return value
    numbers = re.findall(r"(?<![\d.-])\d+(?![\d.])", text)
    if len(set(numbers)) == 1:
        return int(numbers[0])
    raise InferenceError(f"Cannot read a coin count from the model reply: {text[:200]!r}")


class VisionLlmEngine(BaseEngine):
    def __init__(self, model=DEFAULT_MODEL, prompt=DEFAULT_PROMPT, max_image_side=1024,
                 timeout=120, max_retries=5):
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a Gemini model id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        for name, value in {"max_image_side": max_image_side, "timeout": timeout}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError(f"{name} must be a positive number")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.api_key = os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise UnavailableEngineError(f"Set {API_KEY_ENV} to a free key from https://aistudio.google.com/apikey")
        self.model = model
        self.prompt = prompt.strip()
        self.max_image_side = int(max_image_side)
        self.timeout = float(timeout)
        self.max_retries = max_retries

    def _encode(self, image):
        picture = Image.fromarray(image)
        picture.thumbnail((self.max_image_side, self.max_image_side))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=90)
        return picture.size, "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    def _post(self, body):
        request = urllib.request.Request(API_URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code not in RETRY_STATUSES or attempt == self.max_retries:
                    raise InferenceError(f"Gemini API returned HTTP {exc.code}: {detail}") from exc
                # The free tier is rate limited and busy models return 503; honour Retry-After when the provider sends it.
                wait = exc.headers.get("Retry-After") if exc.headers else None
                time.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else 2 ** (attempt + 1))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.max_retries:
                    raise InferenceError(f"Cannot reach the Gemini API: {exc}") from exc
                time.sleep(2 ** (attempt + 1))

    def run(self, image):
        size, data_url = self._encode(image)
        reply = self._post({"model": self.model, "temperature": 0, "messages": [
            {"role": "user", "content": [{"type": "text", "text": self.prompt},
                                         {"type": "image_url", "image_url": {"url": data_url}}]}]})
        if "error" in reply:
            raise InferenceError(f"Gemini API error: {reply['error']}")
        try:
            text = reply["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"Unexpected Gemini API response: {str(reply)[:300]}") from exc
        return CountResult(count=parse_count(text),
                           metadata={"model": self.model, "resolved_model": reply.get("model"), "reply": text,
                                     "usage": reply.get("usage"), "sent_size": list(size)})
