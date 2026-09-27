import io
import json
import os
import unittest
import urllib.error
from email.message import Message
from unittest.mock import patch

import numpy as np

from coincounter import CoinCounter
from coincounter.engines.vision_llm import API_KEY_ENV, VisionLlmEngine, parse_count
from coincounter.exceptions import InferenceError, UnavailableEngineError

# Live calls use the free but rate-limited Gemini tier, so the real-API test is opt-in.
RUN_API = os.environ.get("COINCOUNTER_TEST_VISION_LLM") == "1" and os.environ.get(API_KEY_ENV)


def reply(content, **extra):
    body = {"model": "gemini-3.8-flash",
            "choices": [{"message": {"content": content}}], **extra}
    return io.BytesIO(json.dumps(body).encode())


def http_error(code):
    headers = Message()
    headers["Retry-After"] = "0"
    return urllib.error.HTTPError("url", code, "error", headers, io.BytesIO(b"rate limited"))


class ParseCountTests(unittest.TestCase):
    def test_replies(self):
        for text, count in [('{"count": 7}', 7), ('```json\n{"count": 0}\n```', 0),
                            ("There are 12 coins.", 12), ('{"count": "3"} so 3', 3)]:
            with self.subTest(text=text):
                self.assertEqual(parse_count(text), count)

    def test_unreadable_replies(self):
        for text in ["", "no idea", "between 3 and 5", '{"count": -2}', "about 4.5"]:
            with self.subTest(text=text), self.assertRaises(InferenceError):
                parse_count(text)


@patch.dict(os.environ, {API_KEY_ENV: "test-key"})
class VisionLlmEngineTests(unittest.TestCase):
    def test_missing_api_key(self):
        with patch.dict(os.environ, {API_KEY_ENV: ""}):
            with self.assertRaisesRegex(UnavailableEngineError, API_KEY_ENV):
                VisionLlmEngine()

    def test_invalid_parameters(self):
        for parameters in [{"model": ""}, {"prompt": " "}, {"max_image_side": 0}, {"timeout": True},
                           {"max_retries": -1}, {"max_retries": 1.5}]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                VisionLlmEngine(**parameters)

    @patch("urllib.request.urlopen")
    def test_request_and_result(self, urlopen):
        urlopen.return_value.__enter__.return_value = reply('{"count": 5}', usage={"total_tokens": 300})
        with CoinCounter("vision_llm", {"max_image_side": 64}) as counter:
            result = counter.run(np.zeros((200, 100, 3), dtype=np.uint8))
        self.assertEqual(result.count, 5)
        self.assertIsNone(result.confidence)
        self.assertEqual(result.metadata["sent_size"], [32, 64])
        self.assertEqual(result.metadata["resolved_model"], "gemini-3.8-flash")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "gemini-3.8-flash")
        self.assertEqual(body["temperature"], 0)
        image = body["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image.startswith("data:image/jpeg;base64,"))

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_retries_rate_limit(self, urlopen, sleep):
        success = unittest.mock.MagicMock()
        success.__enter__.return_value = reply('{"count": 2}')
        urlopen.side_effect = [http_error(503), success]
        self.assertEqual(VisionLlmEngine().run(np.zeros((8, 8, 3), dtype=np.uint8)).count, 2)
        sleep.assert_called_once_with(0.0)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_gives_up(self, urlopen, sleep):
        urlopen.side_effect = http_error(429)
        with self.assertRaisesRegex(InferenceError, "429"):
            VisionLlmEngine(max_retries=2).run(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(urlopen.call_count, 3)
        urlopen.reset_mock()
        urlopen.side_effect = http_error(401)
        with self.assertRaisesRegex(InferenceError, "401"):
            VisionLlmEngine().run(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(urlopen.call_count, 1)

    @patch("urllib.request.urlopen")
    def test_error_and_empty_replies(self, urlopen):
        engine = VisionLlmEngine()
        for body in [io.BytesIO(b'{"error": {"message": "no endpoints"}}'), reply(None)]:
            urlopen.return_value.__enter__.return_value = body
            with self.subTest(body=body), self.assertRaises(InferenceError):
                engine.run(np.zeros((8, 8, 3), dtype=np.uint8))


@unittest.skipUnless(RUN_API, f"set COINCOUNTER_TEST_VISION_LLM=1 and {API_KEY_ENV} to call the real API")
class VisionLlmApiTests(unittest.TestCase):
    def test_blank_image(self):
        with CoinCounter("vision_llm") as counter:
            result = counter.run(np.full((240, 240, 3), 255, dtype=np.uint8))
        self.assertIs(type(result.count), int)
        self.assertTrue(result.metadata["reply"])
