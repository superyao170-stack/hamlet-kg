"""Offline transport/configuration tests; no credentials or network required."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from hamletkg.llm import LLMConfig, LLMError, OpenAICompatibleClient


FAKE_KEY = "fake-not-real-transport-test-key"
FAKE_PROXY = "http://127.0.0.1:9876"


class FakeResponse:
    def __init__(self, payload: object, *, raw: bool = False) -> None:
        self.body = (payload if raw else json.dumps(payload).encode("utf-8"))

    def read(self, *args: object) -> bytes:
        return self.body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def completion(content: object = '{"pong": true}', *, finish: str = "stop",
               model: str = "deepseek-flash") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }


def config(**changes: object) -> LLMConfig:
    options = {
        "base_url": "https://api.deepseek.com",
        "api_key": FAKE_KEY,
        "model": "deepseek-flash",
        "timeout_seconds": 7.0,
        "retries": 0,
        "backoff_seconds": 0.0,
    }
    options.update(changes)
    return LLMConfig(**options)


def http_error(status: int, *, body: str | None = None) -> urllib.error.HTTPError:
    # Malicious provider error bodies can echo a credential; never surface them.
    return urllib.error.HTTPError(
        "https://api.deepseek.com/chat/completions", status, "provider error", {},
        io.BytesIO((body or f"echo Bearer {FAKE_KEY}").encode("utf-8")),
    )


class LLMTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.opener = Mock()
        self.opener.open.return_value = FakeResponse(completion())
        self.builder_patch = patch("hamletkg.llm.urllib.request.build_opener",
                                   return_value=self.opener)
        self.builder = self.builder_patch.start()
        self.addCleanup(self.builder_patch.stop)
        self.install_patch = patch("hamletkg.llm.urllib.request.install_opener")
        self.install = self.install_patch.start()
        self.addCleanup(self.install_patch.stop)
        self.network_guard_patch = patch("hamletkg.llm.urllib.request.urlopen",
                                         side_effect=AssertionError("real network disabled in tests"))
        self.network_guard_patch.start()
        self.addCleanup(self.network_guard_patch.stop)
        self.sleep_patch = patch("hamletkg.llm.time.sleep")
        self.sleep = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def test_json_request_has_auth_and_nonthinking_json_contract(self) -> None:
        client = OpenAICompatibleClient(config())
        result = client.complete_json(system="Return JSON, example: {\"pong\":true}",
                                      user="Respond with the JSON object.")
        self.assertEqual({"pong": True}, result)
        request = self.opener.open.call_args.args[0]
        self.assertEqual("POST", request.get_method())
        self.assertEqual("https://api.deepseek.com/chat/completions", request.full_url)
        self.assertEqual(f"Bearer {FAKE_KEY}", request.get_header("Authorization"))
        self.assertEqual("application/json", request.get_header("Content-type"))
        self.assertEqual(7.0, self.opener.open.call_args.kwargs["timeout"])
        payload = json.loads(request.data)
        self.assertEqual("deepseek-flash", payload["model"])
        self.assertEqual({"type": "disabled"}, payload["thinking"])
        self.assertEqual({"type": "json_object"}, payload["response_format"])
        self.assertGreater(payload["max_tokens"], 0)
        self.assertNotIn("extra_body", payload)
        self.assertNotIn(FAKE_KEY, request.data.decode("utf-8"))
        self.install.assert_not_called()

    def test_list_models_uses_get_without_generation(self) -> None:
        self.opener.open.return_value = FakeResponse({
            "object": "list", "data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}],
        })
        client = OpenAICompatibleClient(config())
        self.assertEqual(["deepseek-flash", "deepseek-v4-pro"], client.list_models())
        request = self.opener.open.call_args.args[0]
        self.assertEqual("GET", request.get_method())
        self.assertEqual("https://api.deepseek.com/models", request.full_url)
        self.assertIsNone(request.data)
        self.assertEqual(1, self.opener.open.call_count)

    def test_default_check_only_discovers_models(self) -> None:
        self.opener.open.return_value = FakeResponse({"data": [{"id": "deepseek-flash"}]})
        client = OpenAICompatibleClient(config())
        result = client.check()
        self.assertIsInstance(result, dict)
        self.assertEqual("deepseek-flash", result["requested_model"])
        self.assertIn("deepseek-flash", result["models"])
        self.assertEqual(1, self.opener.open.call_count)
        self.assertTrue(self.opener.open.call_args.args[0].full_url.endswith("/models"))

    def test_explicit_generation_check_sends_minimal_json_prompt(self) -> None:
        self.opener.open.side_effect = [
            FakeResponse({"data": [{"id": "deepseek-flash"}]}),
            FakeResponse(completion('{"ok": true}')),
        ]
        configured = config(max_tokens=2048)
        client = OpenAICompatibleClient(configured)
        result = client.check(generate=True)
        self.assertIsInstance(result, dict)
        self.assertEqual(2, self.opener.open.call_count)
        request = self.opener.open.call_args_list[1].args[0]
        payload = json.loads(request.data)
        self.assertEqual({"type": "json_object"}, payload["response_format"])
        self.assertEqual({"type": "disabled"}, payload["thinking"])
        prompt = " ".join(item["content"] for item in payload["messages"])
        self.assertIn("json", prompt.lower())
        self.assertIn("ok", prompt.lower())
        self.assertGreater(payload["max_tokens"], 0)
        self.assertLessEqual(payload["max_tokens"], 64)
        self.assertEqual(2048, client.config.max_tokens)
        self.assertIs(configured, client.config)

    def test_generation_check_rejects_unexpected_json_schema(self) -> None:
        self.opener.open.side_effect = [
            FakeResponse({"data": [{"id": "deepseek-flash"}]}),
            FakeResponse(completion('{"unexpected": true}')),
        ]
        with self.assertRaises(LLMError):
            OpenAICompatibleClient(config()).check(generate=True)

    def test_explicit_proxy_is_client_local(self) -> None:
        client = OpenAICompatibleClient(config(proxy_url=FAKE_PROXY))
        handlers = self.builder.call_args.args
        proxy_handlers = [handler for handler in handlers if hasattr(handler, "proxies")]
        self.assertEqual(1, len(proxy_handlers))
        self.assertEqual(FAKE_PROXY, proxy_handlers[0].proxies["https"])
        self.assertEqual(FAKE_PROXY, proxy_handlers[0].proxies["http"])
        self.assertIs(self.opener, client.opener)
        self.install.assert_not_called()

    def test_http_status_messages_never_echo_provider_body_or_key(self) -> None:
        for status in (400, 401, 402, 403, 404, 422, 429, 500, 503):
            with self.subTest(status=status):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = http_error(status)
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(LLMError) as raised:
                        OpenAICompatibleClient(config()).complete_json(system="JSON", user="JSON")
                surfaced = str(raised.exception) + stdout.getvalue() + stderr.getvalue()
                self.assertIn(str(status), str(raised.exception))
                self.assertNotIn(FAKE_KEY, surfaced)
                self.assertNotIn("echo Bearer", surfaced)
                self.assertEqual(1, self.opener.open.call_count)

    def test_auth_and_bad_request_errors_are_not_retried(self) -> None:
        for status in (400, 401, 402, 403, 404, 422):
            with self.subTest(status=status):
                self.opener.open.reset_mock()
                self.sleep.reset_mock()
                self.opener.open.side_effect = http_error(status)
                with self.assertRaises(LLMError):
                    OpenAICompatibleClient(config(retries=3, backoff_seconds=.25)).list_models()
                self.assertEqual(1, self.opener.open.call_count)
                self.sleep.assert_not_called()

    def test_retryable_statuses_recover_with_bounded_backoff(self) -> None:
        self.opener.open.side_effect = [http_error(429), http_error(503), FakeResponse(completion())]
        client = OpenAICompatibleClient(config(retries=2, backoff_seconds=.25))
        self.assertEqual({"pong": True}, client.complete_json(system="JSON", user="JSON"))
        self.assertEqual(3, self.opener.open.call_count)
        self.assertEqual([.25, .5], [call.args[0] for call in self.sleep.call_args_list])
        self.assertEqual(3, client.last_response_metadata["attempts"])

    def test_retry_exhaustion_is_bounded_and_sanitized(self) -> None:
        self.opener.open.side_effect = [http_error(500) for _ in range(4)]
        with self.assertRaises(LLMError) as raised:
            OpenAICompatibleClient(config(retries=3, backoff_seconds=.1)).list_models()
        self.assertEqual(4, self.opener.open.call_count)
        self.assertEqual(3, self.sleep.call_count)
        self.assertNotIn(FAKE_KEY, str(raised.exception))

    def test_network_error_exposes_type_without_detail(self) -> None:
        for error in (urllib.error.URLError(f"connection detail {FAKE_KEY}"),
                      TimeoutError(f"timeout detail {FAKE_KEY}")):
            with self.subTest(kind=type(error).__name__):
                self.opener.open.side_effect = error
                with self.assertRaises(LLMError) as raised:
                    OpenAICompatibleClient(config()).list_models()
                self.assertNotIn(FAKE_KEY, str(raised.exception))
                self.assertNotIn("connection detail", str(raised.exception))
                self.assertNotIn("timeout detail", str(raised.exception))

    def test_empty_and_truncated_completion_are_rejected(self) -> None:
        for payload in (completion(""), completion("   "), completion(None),
                        completion('{"pong": true}', finish="length")):
            with self.subTest(content=payload["choices"][0]["message"]["content"],
                              finish=payload["choices"][0]["finish_reason"]):
                self.opener.open.side_effect = None
                self.opener.open.return_value = FakeResponse(payload)
                with self.assertRaises(LLMError):
                    OpenAICompatibleClient(config()).complete_json(system="JSON", user="JSON")

    def test_invalid_json_response_raises_controlled_error(self) -> None:
        self.opener.open.return_value = FakeResponse(b"not-json-secret-" + FAKE_KEY.encode(), raw=True)
        with self.assertRaises(LLMError) as raised:
            OpenAICompatibleClient(config()).list_models()
        self.assertNotIn(FAKE_KEY, str(raised.exception))

    def test_malformed_model_list_is_rejected(self) -> None:
        for payload in ({}, {"data": None}, {"data": ["deepseek-flash"]},
                        {"data": [{"id": 123}]}, {"data": [{"name": "missing-id"}]}):
            with self.subTest(shape=list(payload)):
                self.opener.open.return_value = FakeResponse(payload)
                with self.assertRaises(LLMError):
                    OpenAICompatibleClient(config()).list_models()

    def test_missing_message_and_invalid_content_are_rejected(self) -> None:
        for payload in ({"choices": []}, {}, completion("invalid " + FAKE_KEY)):
            with self.subTest(shape=list(payload)):
                self.opener.open.return_value = FakeResponse(payload)
                with self.assertRaises(LLMError) as raised:
                    OpenAICompatibleClient(config()).complete_json(system="JSON", user="JSON")
                self.assertNotIn(FAKE_KEY, str(raised.exception))

    def test_response_metadata_is_whitelisted_and_model_is_redacted(self) -> None:
        payload = completion(model=f"provider-model-{FAKE_KEY}")
        payload["private_provider_field"] = FAKE_KEY
        payload["usage"]["provider_debug"] = FAKE_KEY
        self.opener.open.return_value = FakeResponse(payload)
        client = OpenAICompatibleClient(config())
        client.complete_json(system="JSON", user="JSON")
        metadata = client.last_response_metadata
        self.assertLessEqual(set(metadata), {"requested_model", "response_model", "usage",
                                             "finish_reason", "attempts"})
        self.assertEqual("deepseek-flash", metadata["requested_model"])
        self.assertNotIn(FAKE_KEY, json.dumps(metadata))
        self.assertNotIn("provider_debug", metadata.get("usage", {}))

    def test_finish_reason_metadata_never_echoes_key(self) -> None:
        self.opener.open.return_value = FakeResponse(completion(finish=FAKE_KEY))
        client = OpenAICompatibleClient(config())
        try:
            client.complete_json(system="JSON", user="JSON")
        except LLMError as error:
            self.assertNotIn(FAKE_KEY, str(error))
        self.assertNotIn(FAKE_KEY, json.dumps(client.last_response_metadata))

    def test_failed_request_clears_previous_response_metadata(self) -> None:
        self.opener.open.side_effect = [FakeResponse(completion()), http_error(401)]
        client = OpenAICompatibleClient(config())
        client.complete_json(system="JSON", user="JSON")
        self.assertIn("response_model", client.last_response_metadata)
        with self.assertRaises(LLMError):
            client.list_models()
        self.assertNotIn("response_model", client.last_response_metadata)

    def test_config_and_client_do_not_mutate_process_settings(self) -> None:
        with patch.dict(os.environ, {"NO_PROXY": "example.invalid", "TEST_SETTING": "unchanged"},
                        clear=True):
            before = dict(os.environ)
            client = OpenAICompatibleClient(config(proxy_url=FAKE_PROXY))
            client.complete_json(system="JSON", user="JSON")
            self.assertEqual(before, dict(os.environ))
            self.install.assert_not_called()


class LLMConfigurationTests(unittest.TestCase):
    def test_repr_redacts_api_key_and_proxy_url(self) -> None:
        configured = config(proxy_url=FAKE_PROXY)
        representation = repr(configured)
        self.assertNotIn(FAKE_KEY, representation)
        self.assertNotIn(FAKE_PROXY, representation)

    def test_invalid_numeric_and_mode_settings_are_rejected(self) -> None:
        invalid = [
            ("timeout_seconds", 0), ("timeout_seconds", -1),
            ("timeout_seconds", float("nan")), ("timeout_seconds", float("inf")),
            ("max_tokens", 0), ("max_tokens", 1.5),
            ("retries", -1), ("retries", 4), ("retries", 1.5),
            ("backoff_seconds", -1), ("backoff_seconds", float("nan")),
            ("backoff_seconds", float("inf")), ("thinking", "sometimes"),
            ("base_url", "file:///tmp/fake"), ("proxy_url", "file:///tmp/fake"),
            ("base_url", "https://api.deepseek.com:invalid"),
            ("proxy_url", "http://tester:fake-password@127.0.0.1:9876"),
        ]
        for name, value in invalid:
            with self.subTest(field=name):
                with self.assertRaises((LLMError, ValueError)):
                    config(**{name: value})

    def test_environment_overrides_dotenv_without_mutating_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text(
                'HAMLET_LLM_BASE_URL="https://api.deepseek.com"\n'
                'HAMLET_LLM_API_KEY="file-fake-key"\n'
                'HAMLET_LLM_MODEL="deepseek-v4-flash"\n'
                'HAMLET_LLM_TIMEOUT_SECONDS=5\n', encoding="utf-8")
            with patch.dict(os.environ, {"HAMLET_LLM_API_KEY": "process-fake-key",
                                         "HAMLET_LLM_MODEL": "deepseek-flash"}, clear=True):
                before = dict(os.environ)
                configured = LLMConfig.from_environment(dotenv_path=dotenv)
                self.assertEqual("process-fake-key", configured.api_key)
                self.assertEqual("deepseek-flash", configured.model)
                self.assertEqual("https://api.deepseek.com", configured.base_url)
                self.assertEqual(5.0, configured.timeout_seconds)
                self.assertEqual(before, dict(os.environ))

    def test_dotenv_shell_syntax_is_literal_and_never_executed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "must-not-exist"
            fake_literal = f"$(touch${{IFS}}{marker})`touch${{IFS}}{marker}`$UNDEFINED_VALUE"
            dotenv = root / ".env"
            dotenv.write_text(
                "# This is data, not a shell program.\n"
                'export HAMLET_LLM_BASE_URL="https://api.deepseek.com"\n'
                f'HAMLET_LLM_API_KEY="{fake_literal}"\n'
                "HAMLET_LLM_MODEL='deepseek-flash'\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                configured = LLMConfig.from_environment(dotenv_path=dotenv)
                self.assertEqual(fake_literal, configured.api_key)
                self.assertEqual("deepseek-flash", configured.model)
                self.assertEqual({}, dict(os.environ))
                self.assertFalse(marker.exists())

    def test_dotenv_preserves_hash_inside_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text(
                '# Comment on a separate line.\n'
                'HAMLET_LLM_BASE_URL="https://api.deepseek.com"\n'
                'HAMLET_LLM_API_KEY="fake#literal"\n'
                "HAMLET_LLM_MODEL=deepseek-flash\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                configured = LLMConfig.from_environment(dotenv_path=dotenv)
                self.assertEqual("fake#literal", configured.api_key)
                self.assertEqual("https://api.deepseek.com", configured.base_url)
                self.assertEqual("deepseek-flash", configured.model)

    def test_dotenv_inline_comments_preserve_quoted_literal_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text(
                'HAMLET_LLM_BASE_URL="https://api.deepseek.com" # endpoint\n'
                'HAMLET_LLM_API_KEY="fake#literal$UNDEFINED_VALUE" # fake key\n'
                "HAMLET_LLM_MODEL=deepseek-flash # current model\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                configured = LLMConfig.from_environment(dotenv_path=dotenv)
                self.assertEqual("fake#literal$UNDEFINED_VALUE", configured.api_key)
                self.assertEqual("https://api.deepseek.com", configured.base_url)
                self.assertEqual("deepseek-flash", configured.model)
                self.assertEqual({}, dict(os.environ))

    def test_default_dotenv_path_is_current_directory_without_env_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "HAMLET_LLM_BASE_URL=https://api.deepseek.com\n"
                "HAMLET_LLM_API_KEY=fake-default-file-key\n"
                "HAMLET_LLM_MODEL=deepseek-flash\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with patch("hamletkg.llm.Path.cwd", return_value=root):
                    configured = LLMConfig.from_environment()
                self.assertEqual("fake-default-file-key", configured.api_key)
                self.assertEqual({}, dict(os.environ))

    def test_missing_settings_raise_without_echoing_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text(f"HAMLET_LLM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(LLMError) as raised:
                    LLMConfig.from_environment(dotenv_path=dotenv)
                self.assertNotIn(FAKE_KEY, str(raised.exception))

    def test_invalid_environment_numbers_raise_controlled_error(self) -> None:
        for variable in ("HAMLET_LLM_TIMEOUT_SECONDS", "HAMLET_LLM_MAX_TOKENS",
                         "HAMLET_LLM_RETRIES", "HAMLET_LLM_BACKOFF_SECONDS"):
            with self.subTest(variable=variable):
                environment = {
                    "HAMLET_LLM_BASE_URL": "https://api.deepseek.com",
                    "HAMLET_LLM_API_KEY": FAKE_KEY,
                    "HAMLET_LLM_MODEL": "deepseek-flash", variable: "invalid-number",
                }
                with tempfile.TemporaryDirectory() as directory:
                    with patch.dict(os.environ, environment, clear=True):
                        with self.assertRaises(LLMError) as raised:
                            LLMConfig.from_environment(dotenv_path=Path(directory) / "absent.env")
                        self.assertNotIn(FAKE_KEY, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
