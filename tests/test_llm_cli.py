from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hamletkg import cli
from hamletkg.detection.policy import DetectionPolicy
from hamletkg.llm import LLMConfig, LLMError


MOCK_KEY = "mock-private-token-never-read-from-environment"
POLICY = {
    "policy_id": "TEST-POLICY", "version": "1.0",
    "rules": [{"rule_id": "R-IN", "operator": "fan_in", "window_seconds": 7200,
               "min_counterparties": 3}],
}


def evidence_payload() -> dict:
    return {"transactions": [{
        "event_id": "TX-TARGET", "observation_time": "2026-10-08T01:01:00Z",
        "score": .8, "candidate_count": 1, "status": "audited",
        "audited_candidates": [{
            "deletion_score_drop": .1, "random_deletion_score_drop": .02,
            "without_evidence_score": .7, "evidence_only_score": .76,
            "graph": {
                "nodes": [
                    {"id": "INJECTED-HUB", "type": "ACCOUNT"},
                    {"id": "NORMAL-SOURCE", "type": "ACCOUNT"},
                    {"id": "NORMAL-DEST", "type": "ACCOUNT"},
                    {"id": "TX-HISTORY", "type": "TRANSACTION", "timestamp": "2026-10-08T01:00:00Z",
                     "source_record_id": "row-1", "attributes": {"amount": 100, "currency": "USD"}},
                    {"id": "TX-TARGET", "type": "TRANSACTION", "timestamp": "2026-10-08T01:01:00Z",
                     "source_record_id": "row-2", "attributes": {"amount": 95, "currency": "USD"}},
                    {"id": "EVENT:1", "type": "BEHAVIOR_EVENT", "motif_type": "pass_through",
                     "anchor_id": "INJECTED-HUB", "span_seconds": 60, "amount_ratio": .95},
                ],
                "edges": [
                    {"source": "NORMAL-SOURCE", "relation": "SENDER_OF", "target": "TX-HISTORY"},
                    {"source": "INJECTED-HUB", "relation": "RECEIVER_OF", "target": "TX-HISTORY"},
                    {"source": "INJECTED-HUB", "relation": "SENDER_OF", "target": "TX-TARGET"},
                    {"source": "NORMAL-DEST", "relation": "RECEIVER_OF", "target": "TX-TARGET"},
                    {"source": "TX-HISTORY", "relation": "MEMBER_OF", "target": "EVENT:1", "member_role": "incoming"},
                    {"source": "TX-TARGET", "relation": "MEMBER_OF", "target": "EVENT:1", "member_role": "outgoing"},
                ],
            },
        }],
    }, {"event_id": "TX-OTHER", "score": .2, "candidate_count": 0,
        "status": "no_event_candidate", "audited_candidates": []}]}


class FakeClient:
    def __init__(self, *, mode: str = "policy", response: dict | None = None,
                 failure: LLMError | None = None) -> None:
        self.config = SimpleNamespace(base_url="https://example.invalid/v1",
                                      model="test-model", api_key=MOCK_KEY)
        self.mode, self.response, self.failure = mode, response, failure
        self.check_calls: list[bool] = []
        self.completion_calls: list[dict] = []
        self.last_response_metadata = {
            "response_model": "actual-response-model", "response_id": "mock-response-id",
            "usage": {"total_tokens": 20}, "attempts": 1,
        }

    def check(self, *, generate: bool = False) -> dict:
        self.check_calls.append(generate)
        if self.failure:
            raise self.failure
        return {"endpoint": self.config.base_url, "requested_model": self.config.model,
                "authentication_verified": True, "models": [self.config.model],
                "configured_model_listed": True, "json_generation_verified": generate}

    def complete_json(self, *, system: str, user: str, **kwargs) -> dict:
        self.completion_calls.append({"system": system, "user": user})
        if self.failure:
            raise self.failure
        if self.response is not None:
            return copy.deepcopy(self.response)
        if self.mode == "narrative":
            facts = json.loads(user)
            return {"event_id": facts["target_event_id"],
                    "selected_fact_ids": [fact["fact_id"] for fact in facts["facts"]],
                    "assessment": "requires_review", "review_actions": ["review_source_records"]}
        return copy.deepcopy(POLICY)


class LLMCLIIntegrationTests(unittest.TestCase):
    def run_cli(self, argv: list[str], client: FakeClient,
                *, expected_env: str, config_failure: LLMError | None = None) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        # Never read a real .env or invoke a real network request in these tests.
        with patch("hamletkg.cli.LLMConfig.from_environment", return_value=client.config,
                   side_effect=config_failure) as loader, \
                patch("hamletkg.cli.OpenAICompatibleClient", return_value=client) as factory, \
                patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network disabled in CLI tests")), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = cli.main(argv)
            except SystemExit as exc:
                code = int(exc.code)
            loader.assert_called_once_with(expected_env)
            if config_failure is None:
                factory.assert_called_once_with(client.config)
            else:
                factory.assert_not_called()
        self.assertNotIn(MOCK_KEY, stdout.getvalue() + stderr.getvalue())
        return int(code), stdout.getvalue(), stderr.getvalue()

    def test_explicit_invalid_environment_file_cannot_fall_back_to_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("missing_argument", "directory_argument", "missing_environment_setting"):
                with self.subTest(kind=kind):
                    source = root if kind == "directory_argument" else root / "missing-settings.env"
                    environment = {"HAMLET_LLM_BASE_URL": "https://example.invalid/v1",
                                   "HAMLET_LLM_API_KEY": MOCK_KEY, "HAMLET_LLM_MODEL": "test-model"}
                    argv = ["llm-check"]
                    if kind == "missing_environment_setting":
                        environment["HAMLET_LLM_ENV_FILE"] = str(source)
                    else:
                        argv.extend(["--env-file", str(source)])
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with patch.dict("os.environ", environment, clear=True), \
                            patch("hamletkg.llm._dotenv_values") as reader, \
                            patch("hamletkg.cli.OpenAICompatibleClient") as factory, \
                            patch("urllib.request.OpenerDirector.open",
                                  side_effect=AssertionError("network disabled in CLI tests")), \
                            contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as failure:
                            cli.main(argv)
                    self.assertEqual(2, failure.exception.code)
                    self.assertEqual("", stdout.getvalue())
                    self.assertIn("explicit LLM environment file is missing", stderr.getvalue())
                    self.assertNotIn(MOCK_KEY, stderr.getvalue())
                    reader.assert_not_called()
                    factory.assert_not_called()

    def test_missing_default_environment_file_allows_explicit_environment_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = {"HAMLET_LLM_BASE_URL": "https://example.invalid/v1",
                           "HAMLET_LLM_API_KEY": MOCK_KEY, "HAMLET_LLM_MODEL": "test-model"}
            with patch.dict("os.environ", environment, clear=True), \
                    patch("hamletkg.llm.Path.cwd", return_value=Path(directory)):
                config = LLMConfig.from_environment()
            self.assertEqual("https://example.invalid/v1", config.base_url)
            self.assertEqual("test-model", config.model)
            self.assertEqual(MOCK_KEY, config.api_key)

    def test_parser_registers_commands_and_env_file_paths(self) -> None:
        parser = cli.build_parser()
        env = "/tmp/nonexistent-test-settings.env"
        checks = [
            (["llm-check", "--env-file", env], cli.command_llm_check),
            (["compile-detection-rules", "--rule", "rule.txt", "--output", "policy.json",
              "--env-file", env], cli.command_compile_detection_rules),
            (["describe-alerts", "--evidence", "evidence.json", "--output", "report",
              "--env-file", env], cli.command_describe_alerts),
        ]
        for argv, handler in checks:
            with self.subTest(command=argv[0]):
                args = parser.parse_args(argv)
                self.assertEqual(env, args.env_file)
                self.assertIs(handler, args.handler)
        self.assertFalse(parser.parse_args(["llm-check"]).generate)
        self.assertEqual(3, parser.parse_args(["describe-alerts", "--evidence", "x", "--output", "y"]).max_transactions)

    def test_llm_check_saves_success_status_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, output = str(root / "unused-settings.env"), root / "status.json"
            client = FakeClient()
            code, stdout, stderr = self.run_cli(["llm-check", "--env-file", env, "--output", str(output)],
                                                client, expected_env=env)
            self.assertEqual(0, code)
            self.assertEqual("", stderr)
            result = json.loads(stdout)
            self.assertTrue(result["ok"])
            self.assertTrue(result["authentication_verified"])
            self.assertFalse(result["json_generation_verified"])
            self.assertEqual(result, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual([False], client.check_calls)
            self.assertEqual([], client.completion_calls)
            self.assertFalse(Path(env).exists())

    def test_llm_check_generate_option_reaches_client(self) -> None:
        client = FakeClient()
        env = "/tmp/nonexistent-test-generation.env"
        code, stdout, _ = self.run_cli(["llm-check", "--env-file", env, "--generate"], client, expected_env=env)
        self.assertEqual(0, code)
        self.assertEqual([True], client.check_calls)
        self.assertTrue(json.loads(stdout)["json_generation_verified"])

    def test_llm_check_authentication_failure_returns_safe_json_and_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, output = str(root / "unused-settings.env"), root / "status.json"
            client = FakeClient(failure=LLMError("LLM HTTP 401: authentication failed"))
            code, stdout, stderr = self.run_cli(["llm-check", "--env-file", env, "--output", str(output)],
                                                client, expected_env=env)
            self.assertEqual(2, code)
            self.assertEqual("", stderr)
            result = json.loads(stdout)
            self.assertFalse(result["ok"])
            self.assertIn("401", result["error"])
            self.assertEqual(result, json.loads(output.read_text(encoding="utf-8")))
            self.assertNotIn("api_key", stdout)

    def test_llm_check_configuration_failure_exits_two_without_success_output(self) -> None:
        env = "/tmp/nonexistent-test-config.env"
        code, stdout, stderr = self.run_cli(["llm-check", "--env-file", env], FakeClient(), expected_env=env,
                                            config_failure=LLMError("required model configuration is missing"))
        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("required model configuration is missing", stderr)

    def test_compile_detection_rules_writes_strict_policy_and_separate_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rule, output, env = root / "rule.txt", root / "policy.json", str(root / "unused-settings.env")
            rule_text = "两小时内从至少三个不同账户汇入资金。\n"
            rule.write_text(rule_text, encoding="utf-8")
            client = FakeClient()
            code, stdout, stderr = self.run_cli(["compile-detection-rules", "--rule", str(rule),
                "--output", str(output), "--samples", "2", "--env-file", env], client, expected_env=env)
            self.assertEqual(0, code)
            self.assertEqual("", stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual({"policy", "compilation"}, set(result))
            self.assertEqual(POLICY, DetectionPolicy.from_dict(result["policy"]).to_dict())
            self.assertEqual({"policy_id", "version", "rules"}, set(result["policy"]))
            quality = result["compilation"]
            self.assertEqual(hashlib.sha256(rule_text.encode("utf-8")).hexdigest(), quality["rule_text_sha256"])
            self.assertEqual(2, quality["requested_samples"])
            self.assertEqual(2, quality["valid_samples"])
            self.assertTrue(quality["requires_domain_review"])
            self.assertEqual("not_verified_by_schema_or_agreement", quality["semantic_correctness"])
            self.assertEqual("actual-response-model", quality["response_metadata"][0]["response_model"])
            self.assertEqual(["fan_in"], json.loads(stdout)["operators"])
            self.assertEqual(2, len(client.completion_calls))

    def test_compile_detection_rules_rejects_unknown_schema_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rule, output, env = root / "rule.txt", root / "policy.json", str(root / "unused-settings.env")
            rule.write_text("两小时内至少三个不同来源。", encoding="utf-8")
            client = FakeClient(response={**POLICY, "label": 1})
            code, stdout, stderr = self.run_cli(["compile-detection-rules", "--rule", str(rule),
                "--output", str(output), "--env-file", env], client, expected_env=env)
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("schema validation", stderr)
            self.assertFalse(output.exists())

    def test_compile_detection_rules_does_not_substitute_defaults_for_unsupported_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rule, output, env = root / "rule.txt", root / "policy.json", str(root / "unused-settings.env")
            rule.write_text("检测包含受制裁国家账户的任意长度资金循环。", encoding="utf-8")
            client = FakeClient(response={"unsupported": True})
            code, stdout, stderr = self.run_cli(["compile-detection-rules", "--rule", str(rule),
                "--output", str(output), "--env-file", env], client, expected_env=env)
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("outside the four supported", stderr)
            self.assertFalse(output.exists())
            self.assertEqual(1, len(client.completion_calls))

    def test_describe_alerts_writes_json_and_identical_markdown_from_existing_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, output, env = root / "evidence.json", root / "report", str(root / "unused-settings.env")
            evidence.write_text(json.dumps(evidence_payload()), encoding="utf-8")
            client = FakeClient(mode="narrative")
            code, stdout, stderr = self.run_cli(["describe-alerts", "--evidence", str(evidence),
                "--output", str(output), "--max-transactions", "1", "--env-file", env], client, expected_env=env)
            self.assertEqual(0, code)
            self.assertEqual("", stderr)
            result = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(result["validated"])
            self.assertEqual("actual-response-model", result["model"])
            self.assertEqual(str(evidence.resolve()), result["evidence_source"]["file"])
            self.assertEqual(hashlib.sha256(evidence.read_bytes()).hexdigest(),
                             result["evidence_source"]["sha256"])
            self.assertEqual(result["markdown"], (output / "report.md").read_text(encoding="utf-8"))
            self.assertEqual(1, result["coverage"]["reported_transactions"])
            self.assertEqual(1, result["coverage"]["omitted_transactions"])
            self.assertTrue(json.loads(stdout)["validated"])
            for observation in result["reports"][0]["observations"]:
                self.assertTrue(observation["transaction_ids"])
                self.assertLessEqual(set(observation["transaction_ids"]), {"TX-HISTORY", "TX-TARGET"})
            self.assertNotIn("INJECTED", client.completion_calls[0]["user"])
            self.assertNotIn("INJECTED", result["markdown"])

    def test_describe_alerts_rejects_fabricated_facts_without_validated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, output, env = root / "evidence.json", root / "report", str(root / "unused-settings.env")
            evidence.write_text(json.dumps(evidence_payload()), encoding="utf-8")
            client = FakeClient(mode="narrative", response={
                "event_id": "TX-TARGET", "selected_fact_ids": ["FABRICATED-FACT"],
                "assessment": "requires_review", "review_actions": [],
            })
            code, stdout, stderr = self.run_cli(["describe-alerts", "--evidence", str(evidence),
                "--output", str(output), "--env-file", env], client, expected_env=env)
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("existing fact identifiers", stderr)
            self.assertFalse((output / "report.json").exists())
            self.assertFalse((output / "report.md").exists())

    def test_describe_alerts_markdown_directory_does_not_publish_success_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, output, env = root / "evidence.json", root / "report", str(root / "unused-settings.env")
            evidence.write_text(json.dumps(evidence_payload()), encoding="utf-8")
            (output / "report.md").mkdir(parents=True)
            client = FakeClient(mode="narrative")
            code, stdout, stderr = self.run_cli(["describe-alerts", "--evidence", str(evidence),
                "--output", str(output), "--env-file", env], client, expected_env=env)
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("unable to write the evidence report files", stderr)
            self.assertTrue((output / "report.md").is_dir())
            self.assertFalse((output / "report.json").exists())
            self.assertFalse((output / ".report.md.pending").exists())
            self.assertFalse((output / ".report.json.pending").exists())

    def test_describe_alerts_staging_write_failure_cleans_pending_and_publishes_no_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, output, env = root / "evidence.json", root / "report", str(root / "unused-settings.env")
            evidence.write_text(json.dumps(evidence_payload()), encoding="utf-8")
            client = FakeClient(mode="narrative")
            original_write = Path.write_text

            def staged_write(path, *args, **kwargs):
                if path.name == ".report.json.pending":
                    raise PermissionError(f"mock write failure containing {MOCK_KEY}")
                return original_write(path, *args, **kwargs)

            with patch.object(Path, "write_text", autospec=True, side_effect=staged_write):
                code, stdout, stderr = self.run_cli(["describe-alerts", "--evidence", str(evidence),
                    "--output", str(output), "--env-file", env], client, expected_env=env)
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("unable to write the evidence report files", stderr)
            self.assertFalse((output / "report.json").exists())
            self.assertFalse((output / "report.md").exists())
            self.assertFalse((output / ".report.md.pending").exists())
            self.assertFalse((output / ".report.json.pending").exists())

    def test_describe_alerts_preserves_user_output_file_and_pending_directory_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("output_file", "pending_directory"):
                with self.subTest(kind=kind):
                    case_root = root / kind
                    case_root.mkdir()
                    evidence, output = case_root / "evidence.json", case_root / "report"
                    env = str(case_root / "unused-settings.env")
                    evidence.write_text(json.dumps(evidence_payload()), encoding="utf-8")
                    if kind == "output_file":
                        output.write_text("existing user file", encoding="utf-8")
                    else:
                        (output / ".report.md.pending").mkdir(parents=True)
                        (output / ".report.md.pending" / "keep.txt").write_text(
                            "existing user directory content", encoding="utf-8")
                    client = FakeClient(mode="narrative")
                    code, stdout, stderr = self.run_cli(["describe-alerts", "--evidence", str(evidence),
                        "--output", str(output), "--env-file", env], client, expected_env=env)
                    self.assertEqual(2, code)
                    self.assertEqual("", stdout)
                    self.assertIn("unable to write the evidence report files", stderr)
                    self.assertFalse((output / "report.json").exists())
                    if kind == "output_file":
                        self.assertEqual("existing user file", output.read_text(encoding="utf-8"))
                    else:
                        self.assertTrue((output / ".report.md.pending").is_dir())
                        self.assertEqual("existing user directory content",
                            (output / ".report.md.pending" / "keep.txt").read_text(encoding="utf-8"))
                        self.assertFalse((output / "report.md").exists())

    def test_describe_alerts_provider_failure_does_not_claim_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, output, env = root / "evidence.json", root / "report", str(root / "unused-settings.env")
            evidence.write_text(json.dumps(evidence_payload()), encoding="utf-8")
            client = FakeClient(mode="narrative", failure=LLMError("LLM HTTP 429: rate limit reached"))
            code, stdout, stderr = self.run_cli(["describe-alerts", "--evidence", str(evidence),
                "--output", str(output), "--env-file", env], client, expected_env=env)
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("429", stderr)
            self.assertFalse((output / "report.json").exists())
            self.assertFalse((output / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
