"""
llm_reply_processor.py

Purpose:
    Parses and validates structured LLM responses for quest-aware NPC turns.
"""

import json
import re
from dataclasses import dataclass

from data_director import QuestPolicy

class StructuredResponseError(RuntimeError):
    """Raised when model output cannot be safely accepted."""

@dataclass(frozen=True)
class ValidatedNpcTurn:
    """A structured NPC turn that passed all policy checks."""
    dialogue: str
    player_intent: str
    topic: str
    policy_id: str
    response_type: str
    clue_claims: tuple[str, ...]
    accepted_clues: tuple[str, ...]
    action_request: None

    def to_skyrim_fields(self) -> dict[str, str]:
        return {
            "response": self.dialogue,
            "intent": self.player_intent,
            "topic": self.topic,
            "policy_id": self.policy_id,
            "claimed_clues": ",".join(self.clue_claims),
            "accepted_clues": ",".join(self.accepted_clues),
            "action": "",
        }

def strip_markdown_fence(raw_text: str) -> str:
    """Robust JSON extraction bypasses markdown and conversational padding."""
    if "```json" in raw_text:
        match = re.search(r'```json(.*?)```', raw_text, re.DOTALL)
        if match: return match.group(1).strip()
    match = re.search(r'\{.*\}', raw_text.strip(), re.DOTALL)
    if not match:
        raise StructuredResponseError("No JSON object found in model output.")
    return match.group(0)

def parse_response_json(raw_text: str) -> dict:
    try:
        parsed = json.loads(raw_text.strip())
        if isinstance(parsed, dict): return parsed
    except json.JSONDecodeError:
        pass
    
    cleaned = strip_markdown_fence(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise StructuredResponseError(f"Model output is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise StructuredResponseError("Model output must be one JSON object.")
    return parsed

def _require_string(data: dict, field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise StructuredResponseError(f"Field '{field}' must be a non-empty string.")
    return value.strip()

def _normalise_clue_claims(data: dict) -> tuple[str, ...]:
    value = data.get("clue_claims")
    if not isinstance(value, list):
        raise StructuredResponseError("Field 'clue_claims' must be a list.")
    claims = []
    for clue_id in value:
        if not isinstance(clue_id, str) or not clue_id.strip():
            raise StructuredResponseError("Every clue_claims entry must be a non-empty string.")
        normalised = clue_id.strip()
        if normalised not in claims:
            claims.append(normalised)
    return tuple(claims)

def _policy_allows_classification(policy: dict, player_intent: str, topic: str) -> None:
    conditions = policy.get("when", {})
    allowed_intents = conditions.get("intents", [])
    allowed_topics = conditions.get("topics", [])
    if player_intent not in allowed_intents:
        raise StructuredResponseError(f"Policy '{policy['policy_id']}' does not allow intent '{player_intent}'.")
    if topic not in allowed_topics:
        raise StructuredResponseError(f"Policy '{policy['policy_id']}' does not allow topic '{topic}'.")

def _policy_allows_state(policy: dict, quest_stage: int, known_clues: set[str]) -> None:
    conditions = policy.get("when", {})
    minimum_stage = conditions.get("quest_stage_min")
    if minimum_stage is not None and quest_stage < minimum_stage:
        raise StructuredResponseError(f"Policy '{policy['policy_id']}' requires quest stage {minimum_stage} or later.")
    clue_not_known = conditions.get("required_clue_not_known")
    if clue_not_known and clue_not_known in known_clues:
        raise StructuredResponseError(f"Policy '{policy['policy_id']}' requires {clue_not_known} to be unknown.")
    clue_known = conditions.get("required_clue_known")
    if clue_known and clue_known not in known_clues:
        raise StructuredResponseError(f"Policy '{policy['policy_id']}' requires {clue_known} to be known.")

def _validate_claims(quest_policy: QuestPolicy, policy: dict, npc_id: str, clue_claims: tuple[str, ...], known_clues: set[str]) -> tuple[str, ...]:
    canonical_clues = set(quest_policy.clues)
    unknown_claims = set(clue_claims).difference(canonical_clues)
    if unknown_claims:
        raise StructuredResponseError("Unknown clue claim(s): " + ", ".join(sorted(unknown_claims)))

    allowed_for_npc = quest_policy.allowed_clues_for_npc(npc_id)
    forbidden_for_npc = quest_policy.forbidden_clues_for_npc(npc_id)

    forbidden_claims = set(clue_claims).intersection(forbidden_for_npc)
    if forbidden_claims:
        raise StructuredResponseError("NPC attempted forbidden clue(s): " + ", ".join(sorted(forbidden_claims)))

    unauthorised_claims = set(clue_claims).difference(allowed_for_npc)
    if unauthorised_claims:
        raise StructuredResponseError("NPC is not authorised for clue(s): " + ", ".join(sorted(unauthorised_claims)))

    policy_awards = set(policy.get("award_clues", []))
    policy_repeatable = set(policy.get("response_directive", {}).get("may_communicate_clues", []))
    permitted_by_policy = policy_awards.union(policy_repeatable)

    claims_outside_policy = set(clue_claims).difference(permitted_by_policy)
    if claims_outside_policy:
        pass # Silently ignore so dialogue can proceed

    return tuple(clue_id for clue_id in clue_claims if clue_id in policy_awards and clue_id not in known_clues)

def validate_structured_response(
    raw_text: str,
    quest_policy: QuestPolicy,
    npc_id: str,
    quest_stage: int,
    known_clues: list[str],
) -> ValidatedNpcTurn:
    data = parse_response_json(raw_text)
    required_fields = set(quest_policy.validation["required_output_fields"])
    missing_fields = sorted(required_fields.difference(data))
    if missing_fields:
        raise StructuredResponseError("Model output is missing field(s): " + ", ".join(missing_fields))

    if data.get("schema_version") != "1.0":
        raise StructuredResponseError("Unsupported schema_version.")

    dialogue = _require_string(data, "dialogue")
    player_intent = _require_string(data, "player_intent")
    topic = _require_string(data, "topic")
    policy_id = _require_string(data, "policy_id")
    response_type = _require_string(data, "response_type")
    clue_claims = _normalise_clue_claims(data)

    if player_intent not in quest_policy.allowed_intents:
        raise StructuredResponseError(f"Unknown player_intent: {player_intent}")
    if topic not in quest_policy.allowed_topics:
        raise StructuredResponseError(f"Unknown topic: {topic}")
    if response_type not in quest_policy.allowed_response_types:
        raise StructuredResponseError(f"Unknown response_type: {response_type}")

    npc_policy_ids = {policy["policy_id"] for policy in quest_policy.policies_for_npc(npc_id)}
    if policy_id not in npc_policy_ids:
        raise StructuredResponseError(f"Policy '{policy_id}' is unavailable to NPC '{npc_id}'.")

    if data.get("action_request") is not None:
        raise StructuredResponseError("action_request must be null for this proof of concept.")

    policy = quest_policy.get_policy(policy_id)

    if response_type != policy.get("response_type"):
        raise StructuredResponseError(f"Policy '{policy_id}' requires response_type '{policy.get('response_type')}'.")

    known_clue_set = set(known_clues)
    _policy_allows_classification(policy, player_intent, topic)
    _policy_allows_state(policy, quest_stage, known_clue_set)
    accepted_clues = _validate_claims(quest_policy, policy, npc_id, clue_claims, known_clue_set)

    return ValidatedNpcTurn(
        dialogue=dialogue,
        player_intent=player_intent,
        topic=topic,
        policy_id=policy_id,
        response_type=response_type,
        clue_claims=clue_claims,
        accepted_clues=accepted_clues,
        action_request=None,
    )
