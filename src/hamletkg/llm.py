"""Optional OpenAI-compatible LLM augmentation.

The LLM may propose atomic event instances and executable risk patterns.  It
does not directly materialize risk events; deterministic validators and graph
matching remain authoritative.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from .models import CaseGraph, RiskPattern
from .patterns import PatternValidationError, validate_pattern, validate_pattern_payload


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    ca_file: str | None = None
    thinking: str = "disabled"
    max_tokens: int = 2048

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        base_url = os.environ.get("HAMLET_LLM_BASE_URL", "").rstrip("/")
        api_key = os.environ.get("HAMLET_LLM_API_KEY", "")
        model = os.environ.get("HAMLET_LLM_MODEL", "")
        if not base_url or not api_key or not model:
            raise LLMError(
                "HAMLET_LLM_BASE_URL, HAMLET_LLM_API_KEY and HAMLET_LLM_MODEL are required"
            )
        configured_ca = os.environ.get("HAMLET_LLM_CA_FILE", "").strip()
        default_ca = "/etc/ssl/cert.pem"
        ca_file = configured_ca or (default_ca if os.path.exists(default_ca) else None)
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=float(os.environ.get("HAMLET_LLM_TIMEOUT_SECONDS", "60")),
            ca_file=ca_file,
            thinking=os.environ.get("HAMLET_LLM_THINKING", "disabled"),
            max_tokens=int(os.environ.get("HAMLET_LLM_MAX_TOKENS", "2048")),
        )


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_environment()

    def complete_json(self, *, system: str, user: str) -> Any:
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "thinking": {"type": self.config.thinking},
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            ssl_context = ssl.create_default_context(cafile=self.config.ca_file)
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
                context=ssl_context,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response does not contain message content") from exc
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
        payload["case_id"] = case_id
        payload.setdefault("risk_events", [])
        payload.setdefault("atomic_relations", [])
        payload.setdefault("risk_relations", [])
        return CaseGraph.from_dict(payload)

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
