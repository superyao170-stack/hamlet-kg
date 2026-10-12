"""Optional JSON LLM client, factual extraction and bounded rule compilation.

Detection policies configure deterministic operators and are frozen before
training. Evidence reports use a separate fact-selection validator. Neither
workflow allows the LLM to assign transaction labels or certify laundering.
"""

from __future__ import annotations

import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .models import CaseGraph, RiskPattern
from .patterns import PatternValidationError, validate_pattern, validate_pattern_payload


class LLMError(RuntimeError):
    pass


def _dotenv_values(path: str | Path) -> dict[str, str]:
    """Read a small literal key/value file without executing shell syntax."""
    source = Path(path)
    if not source.exists():
        return {}
    result = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", stripped)
        if not match:
            raise LLMError(f"invalid .env key/value syntax at line {number}")
        name, value = match.groups()
        if name.startswith("HAMLET_LLM_"):
            if value and value[0] in "\"'":
                quote = value[0]
                end = next((position for position in range(1, len(value))
                            if value[position] == quote and value[position - 1] != "\\"), None)
                if end is None or (value[end + 1:].strip() and not value[end + 1:].strip().startswith("#")):
                    raise LLMError(f"invalid quoted .env value at line {number}")
                value = value[1:end]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
            result[name] = value
    return result


def _valid_url(value: str, name: str) -> None:
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        raise LLMError(f"{name} must have a valid host and port") from None
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LLMError(f"{name} must be an HTTP(S) URL without embedded credentials, query or fragment")


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float = 60.0
    ca_file: str | None = None
    thinking: str = "disabled"
    max_tokens: int = 2048
    proxy_url: str | None = field(default=None, repr=False)
    retries: int = 2
    backoff_seconds: float = .5

    def __post_init__(self) -> None:
        _valid_url(self.base_url, "base_url")
        if self.proxy_url:
            _valid_url(self.proxy_url, "proxy_url")
        if not self.api_key or not self.model or any(char.isspace() for char in self.api_key + self.model):
            raise LLMError("API key and model must be nonempty without whitespace")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise LLMError("timeout_seconds must be finite and positive")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise LLMError("max_tokens must be a positive integer")
        if self.thinking not in ("disabled", "enabled"):
            raise LLMError("thinking must be disabled or enabled")
        if isinstance(self.retries, bool) or not isinstance(self.retries, int) or not 0 <= self.retries <= 3:
            raise LLMError("retries must be an integer between 0 and 3")
        if not math.isfinite(self.backoff_seconds) or self.backoff_seconds < 0:
            raise LLMError("backoff_seconds must be finite and nonnegative")

    @classmethod
    def from_environment(cls, dotenv_path: str | Path | None = None) -> "LLMConfig":
        explicit_file = dotenv_path or os.environ.get("HAMLET_LLM_ENV_FILE")
        file_path = explicit_file or Path.cwd() / ".env"
        if explicit_file and not Path(file_path).is_file():
            raise LLMError("explicit LLM environment file is missing or is not a regular file")
        try:
            values = _dotenv_values(file_path)
        except (OSError, UnicodeDecodeError):
            raise LLMError("unable to read the LLM environment file") from None
        values.update({name: value for name, value in os.environ.items() if name.startswith("HAMLET_LLM_")})
        base_url = values.get("HAMLET_LLM_BASE_URL", "").strip().rstrip("/")
        api_key = values.get("HAMLET_LLM_API_KEY", "").strip()
        model = values.get("HAMLET_LLM_MODEL", "").strip()
        if not base_url or not api_key or not model:
            raise LLMError(
                "HAMLET_LLM_BASE_URL, HAMLET_LLM_API_KEY and HAMLET_LLM_MODEL are required"
            )
        configured_ca = values.get("HAMLET_LLM_CA_FILE", "").strip()
        default_ca = "/etc/ssl/cert.pem"
        ca_file = configured_ca or (default_ca if os.path.exists(default_ca) else None)
        try:
            return cls(
                base_url=base_url, api_key=api_key, model=model,
                timeout_seconds=float(values.get("HAMLET_LLM_TIMEOUT_SECONDS", "60")),
                ca_file=ca_file, thinking=values.get("HAMLET_LLM_THINKING", "disabled"),
                max_tokens=int(values.get("HAMLET_LLM_MAX_TOKENS", "2048")),
                proxy_url=values.get("HAMLET_LLM_PROXY_URL") or None,
                retries=int(values.get("HAMLET_LLM_RETRIES", "2")),
                backoff_seconds=float(values.get("HAMLET_LLM_BACKOFF_SECONDS", ".5")),
            )
        except (ValueError, OverflowError) as exc:
            raise LLMError("LLM environment contains an invalid numeric setting") from exc


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_environment()
        proxies = ({"http": self.config.proxy_url, "https": self.config.proxy_url}
                   if self.config.proxy_url else urllib.request.getproxies())
        try:
            context = ssl.create_default_context(cafile=self.config.ca_file)
        except (OSError, ssl.SSLError) as exc:
            raise LLMError("unable to load the configured TLS certificate bundle") from exc
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(proxies), urllib.request.HTTPSHandler(context=context))
        self.last_response_metadata: dict[str, Any] = {}

    def _redact(self, value: str) -> str:
        return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value.replace(self.config.api_key, "[REDACTED]"))

    def _request_json(self, path: str, *, payload: dict | None = None) -> dict:
        self.last_response_metadata = {}
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        for attempt in range(self.config.retries + 1):
            try:
                with self.opener.open(request, timeout=self.config.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise LLMError("LLM HTTP response must be a JSON object")
                self.last_response_metadata = {"attempts": attempt + 1}
                return result
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if (status == 429 or 500 <= status < 600) and attempt < self.config.retries:
                    time.sleep(self.config.backoff_seconds * 2 ** attempt)
                    continue
                reasons = {400: "invalid request or unsupported model", 401: "authentication failed",
                           402: "insufficient balance", 403: "access denied", 404: "endpoint or model not found",
                           429: "rate limit reached"}
                reason = reasons.get(status, "provider unavailable" if status >= 500 else "request rejected")
                # Never echo provider bodies, headers, URL credentials or keys.
                raise LLMError(f"LLM HTTP {status}: {reason}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise LLMError(f"LLM transport failed ({type(exc).__name__}); check endpoint, TLS and proxy") from None
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise LLMError("LLM HTTP response is not valid UTF-8 JSON") from None
        raise LLMError("LLM request exhausted retry attempts")

    def list_models(self) -> list[str]:
        result = self._request_json("/models")
        rows = result.get("data")
        if not isinstance(rows, list) or any(not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in rows):
            raise LLMError("model list response is missing valid model identifiers")
        return sorted({self._redact(item["id"]) for item in rows})

    def check(self, *, generate: bool = False) -> dict:
        models = self.list_models()
        result = {"endpoint": self.config.base_url, "requested_model": self.config.model,
                  "authentication_verified": True, "models": models,
                  "configured_model_listed": self.config.model in models,
                  "json_generation_verified": False}
        if generate:
            payload = self.complete_json(system='Return only JSON: {"ok": true}.',
                                         user='Connectivity test. Return the JSON object {"ok": true}.',
                                         max_tokens=min(64, self.config.max_tokens))
            if payload != {"ok": True}:
                raise LLMError("JSON connectivity response did not match the requested schema")
            result["json_generation_verified"] = True
            result["response_metadata"] = dict(self.last_response_metadata)
        return result

    def complete_json(self, *, system: str, user: str, max_tokens: int | None = None) -> Any:
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
            raise LLMError("max_tokens override must be a positive integer")
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "thinking": {"type": self.config.thinking},
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        result = self._request_json("/chat/completions", payload=payload)
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response does not contain message content") from exc
        if choice.get("finish_reason") == "length":
            raise LLMError("LLM output was truncated; increase max_tokens or reduce input")
        if content is None or (isinstance(content, str) and not content.strip()):
            raise LLMError("LLM returned empty content")
        usage = result.get("usage") or {}
        self.last_response_metadata.update({
            "requested_model": self.config.model,
            "response_model": self._redact(str(result.get("model", "unknown"))),
            "finish_reason": self._redact(str(choice.get("finish_reason"))) if choice.get("finish_reason") is not None else None,
            "usage": {name: usage[name] for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                      if isinstance(usage, dict) and isinstance(usage.get(name), int)},
        })
        return _decode_json_content(content)


def _decode_json_content(content: str | dict | list) -> Any:
    if isinstance(content, (dict, list)):
        return content
    cleaned = str(content).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM returned invalid JSON") from exc


class LLMAugmentor:
    def __init__(self, client: OpenAICompatibleClient | None = None) -> None:
        self.client = client or OpenAICompatibleClient()

    def extract_case(self, *, case_id: str, text: str) -> CaseGraph:
        system = (
            "You extract only explicit atomic actions from financial case text. "
            "Never infer a risk event or a criminal conclusion. Return one JSON object "
            "with entities and atomic_events. Each event needs event_id, action_type, "
            "roles, ISO-8601 timestamp, attributes, and optional source_record_id."
        )
        payload = self.client.complete_json(system=system, user=text)
        if not isinstance(payload, dict):
            raise LLMError("atomic extraction must return a JSON object")
        allowed = {"case_id", "entities", "atomic_events", "risk_events", "atomic_relations", "risk_relations", "metadata"}
        if set(payload) - allowed:
            raise LLMError("atomic extraction contains unsupported top-level fields")
        for name in ("risk_events", "atomic_relations", "risk_relations"):
            if name in payload and not isinstance(payload[name], list):
                raise LLMError("atomic extraction optional risk and relation fields must be empty lists")
        if "metadata" in payload and not isinstance(payload["metadata"], dict):
            raise LLMError("atomic extraction optional metadata must be an empty object")
        if any(payload.get(name) for name in ("risk_events", "atomic_relations", "risk_relations", "metadata")):
            raise LLMError("atomic extraction may not inject risk events, graph relations or evaluation metadata")
        forbidden = {"islaundering", "isfraud", "fraudstep", "label", "labels", "evaluationlabels"}

        def contains_label(value: Any) -> bool:
            if isinstance(value, dict):
                return any(re.sub(r"[^a-z0-9]", "", str(name).lower()) in forbidden
                           or contains_label(item) for name, item in value.items())
            return isinstance(value, list) and any(contains_label(item) for item in value)

        for collection in ("entities", "atomic_events"):
            if not isinstance(payload.get(collection), list):
                raise LLMError(f"atomic extraction requires a {collection} list")
            ids = []
            id_field = "entity_id" if collection == "entities" else "event_id"
            for item in payload[collection]:
                if (not isinstance(item, dict) or not isinstance(item.get(id_field), str)
                        or not item[id_field].strip() or item[id_field] != item[id_field].strip()):
                    raise LLMError("atomic extraction contains invalid entity/event identifiers")
                allowed_fields = ({"entity_id", "entity_type", "attributes"} if collection == "entities"
                                  else {"event_id", "action_type", "roles", "timestamp", "attributes", "source_record_id"})
                if set(item) - allowed_fields:
                    raise LLMError("atomic extraction contains unsupported entity/event fields")
                ids.append(item[id_field])
                attributes = item.get("attributes", {})
                if not isinstance(attributes, dict) or contains_label(attributes):
                    raise LLMError("atomic extraction may not inject label attributes")
            if len(ids) != len(set(ids)):
                raise LLMError("atomic extraction contains duplicate identifiers")
        payload["case_id"] = case_id
        payload.setdefault("risk_events", [])
        payload.setdefault("atomic_relations", [])
        payload.setdefault("risk_relations", [])
        try:
            return CaseGraph.from_dict(payload)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise LLMError("atomic extraction failed entity, event or reference validation") from None

    def compile_detection_policy(
        self, rule_text: str, *, samples: int = 1, max_repair_attempts: int = 1,
    ) -> tuple[Any, dict]:
        """Compile to a bounded operator policy, not arbitrary executable code.

        Syntactic validity and repeated agreement do not establish semantic
        correctness. The generated policy is frozen before model training.
        """
        from .detection.policy import DetectionPolicy

        if (isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 5
                or isinstance(max_repair_attempts, bool) or not isinstance(max_repair_attempts, int)
                or not 0 <= max_repair_attempts <= 1):
            raise ValueError("policy samples must be 1..5 and repair attempts 0..1")
        system = (
            'Compile the supplied AML behavior descriptions to exactly one JSON detection policy. '
            'Top-level keys are policy_id, version, rules. No description, quality, label or extra fields. '
            'Each rule requires rule_id, operator, window_seconds. The only operators are '
            'fan_in, fan_out, pass_through, relay. fan_in/fan_out additionally allow only '
            'min_counterparties (integer >=2). pass_through additionally allows only '
            'min_incoming (integer >=1), ratio_min, ratio_max. relay additionally allows only '
            'ratio_min, ratio_max: these ratios apply to EACH intermediate account, not the whole chain. '
            'relay always means three chronological transfers and four distinct accounts; arbitrary '
            'cycles, paths, countries, sanctions, thresholds on individual amounts and unsupported '
            'predicates cannot be represented. If the text requests any unsupported predicate, return '
            '{"unsupported": true}; do not omit it or silently approximate it. '
            'All ratios are positive and ratio_min <= ratio_max. Preserve supplied numbers exactly '
            'while converting hours/minutes to seconds and percentages to ratios. Never infer a '
            'laundering conclusion or generate a transaction label. Rules are candidate event definitions. '
            'Example JSON: {"policy_id":"P-DEMO","version":"1.0","rules":'
            '[{"rule_id":"R1","operator":"pass_through","window_seconds":86400,'
            '"min_incoming":1,"ratio_min":0.8,"ratio_max":1.2}]}. '
            'The current bounded implementation uses strictly earlier history plus the current '
            'transaction, and aggregates all retained eligible history, not a combinatorial subset.'
        )
        candidates = []
        repairs = 0
        responses = []
        for _ in range(samples):
            request = rule_text
            for attempt in range(max_repair_attempts + 1):
                if attempt:
                    repairs += 1
                try:
                    payload = self.client.complete_json(system=system, user=request)
                    if isinstance(payload, dict) and payload.get("unsupported") is True:
                        raise LLMError("rule requests predicates outside the four supported detection operators")
                    policy = DetectionPolicy.from_dict(payload)
                    candidates.append(policy)
                    responses.append(dict(getattr(self.client, "last_response_metadata", {})))
                    break
                except ValueError as exc:
                    if attempt == max_repair_attempts:
                        break
                    else:
                        request = f"Original rule:\n{rule_text}\nThe candidate failed validation: {type(exc).__name__}. Return corrected JSON within the exact schema."
                except LLMError:
                    # Provider failures and unsupported predicates are never
                    # silently replaced by handcrafted/default policies.
                    raise
        if not candidates:
            raise LLMError("all detection policy candidates failed schema validation")

        def fingerprint(policy):
            rules = [{key: value for key, value in rule.to_dict().items() if key != "rule_id"}
                     for rule in policy.rules]
            return json.dumps(sorted(rules, key=lambda item: json.dumps(item, sort_keys=True)), sort_keys=True)

        counts = Counter(fingerprint(policy) for policy in candidates)
        selected, votes = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0]
        policy = next(item for item in candidates if fingerprint(item) == selected)
        quality = {
            "source": "llm_detection_policy_compilation", "requested_samples": samples,
            "valid_samples": len(candidates), "valid_sample_rate": len(candidates) / samples,
            "semantic_agreement": votes / len(candidates), "repair_attempts": repairs,
            "semantic_correctness": "not_verified_by_schema_or_agreement",
            "requires_domain_review": True, "response_metadata": responses,
        }
        return policy, quality

    def compile_pattern(
        self,
        rule_text: str,
        *,
        max_repair_attempts: int = 1,
        samples: int = 1,
    ) -> RiskPattern:
        if samples < 1:
            raise ValueError("samples must be positive")
        system = (
            "Compile an AML rule into exactly one HAMLET-KG JSON object. "
            "Required top-level keys: pattern_id, risk_type, description, version, "
            "clauses, constraints. Each clause must use exactly these keys: clause_id, "
            "action_type, anchor_role, min_count, and optionally max_count or "
            "distinct_role. Use action_type TRANSFER for both incoming and outgoing "
            "transfers. A TRANSFER has roles sender and receiver; express direction "
            "with anchor_role, not with a new action type. Each constraint must use "
            "exactly the keys constraint_type and parameters. ordered_within parameters "
            "are earlier, later, min_seconds, max_seconds. sum_ratio parameters are "
            "numerator, denominator, attribute, minimum, and optional maximum. "
            "total_amount parameters are clause, attribute, and optional minimum or "
            "maximum. duration_max parameters contain max_seconds. attribute_range "
            "parameters are clause, attribute, and optional minimum or maximum; it "
            "selects eligible events before min_count is checked. role_overlap parameters "
            "are left_clause, left_role, right_clause, right_role, and optional minimum. "
            "Time values must be "
            "numeric seconds. risk_type must copy the narrow business pattern label from "
            "the rule in its original language (especially a quoted label); never replace "
            "it with a broad crime category such as money_laundering. Preserve all numeric "
            "thresholds exactly. Do not add wrapper keys, comments, operators, or conditions "
            "absent from the rule."
        )
        candidates: list[RiskPattern] = []
        repair_attempts = 0
        errors: list[str] = []
        for _ in range(samples):
            try:
                pattern, repairs = self._compile_one(
                    rule_text,
                    system=system,
                    max_repair_attempts=max_repair_attempts,
                )
                candidates.append(pattern)
                repair_attempts += repairs
            except LLMError as exc:
                errors.append(str(exc))
        if not candidates:
            raise LLMError(
                "all LLM pattern samples failed deterministic validation: "
                + "; ".join(errors)
            )
        fingerprints = [self._semantic_fingerprint(item) for item in candidates]
        counts = Counter(fingerprints)
        selected_fingerprint, agreement_count = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        selected = next(
            pattern
            for pattern, fingerprint in zip(candidates, fingerprints)
            if fingerprint == selected_fingerprint
        )
        config = getattr(self.client, "config", None)
        quality = dict(selected.quality)
        quality.update(
            {
                "source": "llm_compilation",
                "model": getattr(config, "model", "unknown"),
                "confidence_status": "uncalibrated",
                "requested_samples": samples,
                "valid_samples": len(candidates),
                "valid_sample_rate": len(candidates) / samples,
                "semantic_agreement": agreement_count / len(candidates),
                "overall_agreement": agreement_count / samples,
                "semantic_variant_count": len(counts),
                "selected_variant_votes": agreement_count,
                "repair_attempts": repair_attempts,
            }
        )
        return replace(selected, quality=quality)

    def _compile_one(
        self,
        rule_text: str,
        *,
        system: str,
        max_repair_attempts: int,
    ) -> tuple[RiskPattern, int]:
        request_text = rule_text
        last_error: Exception | None = None
        for attempt in range(max_repair_attempts + 1):
            payload = self.client.complete_json(system=system, user=request_text)
            try:
                if not isinstance(payload, dict):
                    raise PatternValidationError(
                        "pattern compilation must return a JSON object"
                    )
                validate_pattern_payload(payload)
                pattern = RiskPattern.from_dict(payload)
                validate_pattern(pattern)
                return pattern, attempt
            except (KeyError, TypeError, ValueError, PatternValidationError) as exc:
                last_error = exc
                if attempt >= max_repair_attempts:
                    break
                request_text = (
                    "Original AML rule:\n"
                    f"{rule_text}\n\n"
                    "The previous candidate failed deterministic validation with this "
                    f"error: {exc}. Return a corrected JSON object only."
                )
        raise LLMError(f"LLM pattern failed deterministic validation: {last_error}")

    @staticmethod
    def _semantic_fingerprint(pattern: RiskPattern) -> str:
        clauses = sorted(
            pattern.clauses,
            key=lambda item: (
                item.action_type,
                item.anchor_role,
                item.min_count,
                item.max_count if item.max_count is not None else -1,
                item.distinct_role or "",
                item.clause_id,
            ),
        )
        canonical_ids = {
            clause.clause_id: f"c{index}" for index, clause in enumerate(clauses)
        }
        canonical_clauses = [
            {
                "clause_id": canonical_ids[item.clause_id],
                "action_type": item.action_type,
                "anchor_role": item.anchor_role,
                "min_count": item.min_count,
                "max_count": item.max_count,
                "distinct_role": item.distinct_role,
            }
            for item in clauses
        ]
        clause_reference_keys = {
            "earlier", "later", "numerator", "denominator", "clause",
            "left_clause", "right_clause"
        }
        canonical_constraints = []
        for constraint in pattern.constraints:
            parameters = {
                key: (
                    canonical_ids.get(str(value), str(value))
                    if key in clause_reference_keys
                    else value
                )
                for key, value in constraint.parameters.items()
            }
            canonical_constraints.append(
                {
                    "constraint_type": constraint.constraint_type,
                    "parameters": parameters,
                }
            )
        canonical_constraints.sort(
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        value = {
            "risk_type": re.sub(r"\s+", "", pattern.risk_type).lower(),
            "clauses": canonical_clauses,
            "constraints": canonical_constraints,
        }
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
