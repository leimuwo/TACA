import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from thought_action_retrieval.inference.client import (
    AuthenticationError,
    InferenceClient,
)


def _test_credential():
    return "unit-" + "test-credential"


class _Handler(BaseHTTPRequestHandler):
    response_status = 200
    response_statuses = []
    response_body = {
        "choices": [{"message": {"content": '{"negative_cases": []}'}}]
    }
    requests = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": json.loads(body.decode("utf-8")),
            }
        )
        encoded = json.dumps(type(self).response_body).encode("utf-8")
        status = (
            type(self).response_statuses.pop(0)
            if type(self).response_statuses
            else type(self).response_status
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


class InferenceClientTests(unittest.TestCase):
    def setUp(self):
        _Handler.response_status = 200
        _Handler.response_statuses = []
        _Handler.response_body = {
            "choices": [{"message": {"content": '{"negative_cases": []}'}}]
        }
        _Handler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_posts_openai_messages_and_parses_json_content(self):
        client = InferenceClient(
            endpoint=self.endpoint,
            api_key=_test_credential(),
            request_path="/generate",
            model="pilot-model",
            timeout=5,
        )

        result = client.generate_json(
            system_prompt="system",
            user_prompt="user",
        )

        self.assertEqual(result, {"negative_cases": []})
        request = _Handler.requests[0]
        self.assertEqual(request["path"], "/generate")
        self.assertEqual(request["authorization"], f"Bearer {_test_credential()}")
        self.assertEqual(request["body"]["model"], "pilot-model")
        self.assertEqual(
            request["body"]["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
        )
        self.assertNotIn("response_format", request["body"])

    def test_can_enable_json_response_format_for_compatible_endpoints(self):
        client = InferenceClient(
            endpoint=self.endpoint,
            api_key=_test_credential(),
            json_response_format=True,
            timeout=5,
        )

        client.generate_json(system_prompt="s", user_prompt="u")

        self.assertEqual(
            _Handler.requests[0]["body"]["response_format"],
            {"type": "json_object"},
        )

    def test_bypasses_inherited_proxy_configuration(self):
        client = InferenceClient(
            endpoint=self.endpoint,
            api_key=_test_credential(),
            request_path="/",
            timeout=5,
        )

        with mock.patch.dict(
            os.environ,
            {"HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""},
            clear=False,
        ):
            result = client.generate_json(system_prompt="s", user_prompt="u")

        self.assertEqual(result, {"negative_cases": []})

    def test_authentication_error_does_not_include_secret(self):
        _Handler.response_status = 401
        _Handler.response_body = {"error": "denied"}
        client = InferenceClient(
            endpoint=self.endpoint,
            api_key=_test_credential(),
            timeout=5,
        )

        with self.assertRaises(AuthenticationError) as captured:
            client.preflight()

        self.assertNotIn(_test_credential(), str(captured.exception))
        self.assertIn("authentication", str(captured.exception).lower())

    def test_rejects_non_json_model_content(self):
        _Handler.response_body = {
            "choices": [{"message": {"content": "not json"}}]
        }
        client = InferenceClient(
            endpoint=self.endpoint,
            api_key=_test_credential(),
            timeout=5,
        )

        with self.assertRaisesRegex(ValueError, "valid JSON"):
            client.generate_json(system_prompt="s", user_prompt="u")

    def test_retries_transient_server_errors_with_a_fixed_attempt_cap(self):
        _Handler.response_statuses = [500, 502, 200]
        client = InferenceClient(
            endpoint=self.endpoint,
            api_key=_test_credential(),
            timeout=5,
            retries=2,
        )

        result = client.generate_json(system_prompt="s", user_prompt="u")

        self.assertEqual(result, {"negative_cases": []})
        self.assertEqual(len(_Handler.requests), 3)


if __name__ == "__main__":
    unittest.main()
