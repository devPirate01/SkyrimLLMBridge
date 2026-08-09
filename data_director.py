"""
data_director.py

Purpose:
    The single source of truth for all local file I/O.
    Handles Quest Policies, Profiles, Session Memory, and Logging.
"""

import csv
import json
from pathlib import Path

BRIDGE_DIRECTORY = Path(__file__).resolve().parent
PROFILES_DIRECTORY = BRIDGE_DIRECTORY / "profiles"
MEMORY_DIRECTORY = BRIDGE_DIRECTORY / "memory"
LOGS_DIRECTORY = BRIDGE_DIRECTORY / "logs"
QUESTS_DIRECTORY = BRIDGE_DIRECTORY / "quests"
DEFAULT_POLICY_PATH = QUESTS_DIRECTORY / "poisoned_mead.json"

MAX_RECENT_EXCHANGES = 10

class DataError(RuntimeError):
    """Raised when data loading fails."""

def safe_name(value: str) -> str:
    """Convert a display name or session ID into a safe file name."""
    result = []
    for character in value.lower():
        if character.isalnum():
            result.append(character)
        elif character in {" ", "-", "_"}:
            result.append("_")
    cleaned = "".join(result)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"

def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON through a temporary file so Skyrim never reads half a file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)

# ----------------------------------------------------------------------------
# QUEST POLICY
# ----------------------------------------------------------------------------
class QuestPolicy:
    """Read-only access to one validated quest policy file."""

    def __init__(self, policy_path: Path = DEFAULT_POLICY_PATH):
        self.policy_path = policy_path
        self.data = self._load_policy()

        self.quest = self.data["quest"]
        self.vocabulary = self.data["vocabulary"]
        self.clues = self.data["clues"]
        self.npc_knowledge = self.data["npc_knowledge"]
        self.policies = self.data["policies"]
        self.fallbacks = self.data["fallbacks"]
        self.validation = self.data["validation"]

        self._policies_by_id = {
            policy["policy_id"]: policy
            for policy in self.policies
        }

    def _load_policy(self) -> dict:
        if not self.policy_path.exists():
            raise DataError(f"Quest policy file not found: {self.policy_path}")
        try:
            data = json.loads(self.policy_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise DataError(f"Could not load quest policy: {error}") from error
        return data

    @property
    def quest_id(self) -> str:
        return self.quest["quest_id"]
    @property
    def allowed_intents(self) -> set[str]:
        return set(self.vocabulary["intents"])
    @property
    def allowed_topics(self) -> set[str]:
        return set(self.vocabulary["topics"])
    @property
    def allowed_response_types(self) -> set[str]:
        return set(self.vocabulary["response_types"])
    @property
    def allowed_policy_ids(self) -> set[str]:
        return set(self._policies_by_id)

    def get_policy(self, policy_id: str) -> dict:
        return self._policies_by_id[policy_id]

    def policies_for_npc(self, npc_id: str) -> list[dict]:
        return [p for p in self.policies if npc_id in p.get("npc_ids", [])]

    def allowed_clues_for_npc(self, npc_id: str) -> set[str]:
        return set(self.npc_knowledge.get(npc_id, {}).get("allowed_clues", []))

    def forbidden_clues_for_npc(self, npc_id: str) -> set[str]:
        return set(self.npc_knowledge.get(npc_id, {}).get("forbidden_clues", []))

    def get_fallback(self, fallback_id: str) -> dict:
        return self.fallbacks[fallback_id]

# ----------------------------------------------------------------------------
# PROFILES
# ----------------------------------------------------------------------------
def load_profile(npc_name: str) -> dict:
    """Load one NPC profile."""
    path = PROFILES_DIRECTORY / f"{safe_name(npc_name)}.json"
    if not path.exists():
        raise DataError(f"NPC profile not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------
# MEMORY (With Session Isolation)
# ----------------------------------------------------------------------------
def memory_path(npc_name: str, session_id: str) -> Path:
    """Memory path now isolates by session_id!"""
    return MEMORY_DIRECTORY / f"{safe_name(npc_name)}_{safe_name(session_id)}_history.json"

def load_memory(npc_name: str, session_id: str) -> list[dict]:
    path = memory_path(npc_name, session_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = data.get("messages", [])
        return messages[-(MAX_RECENT_EXCHANGES * 2):]
    except Exception as error:
        print(f"Warning: memory reset for {npc_name} (session {session_id}): {error}")
        return []

def save_exchange(npc_name: str, session_id: str, player_input: str, npc_response: str) -> None:
    messages = load_memory(npc_name, session_id)
    messages.extend([
        {"role": "user", "content": player_input},
        {"role": "assistant", "content": npc_response},
    ])
    messages = messages[-(MAX_RECENT_EXCHANGES * 2):]
    if messages and messages[0]["role"] == "assistant":
        messages = messages[1:]

    write_json_atomic(
        memory_path(npc_name, session_id),
        {"npc_name": npc_name, "session_id": session_id, "messages": messages},
    )

# ----------------------------------------------------------------------------
# CSV LOGGING
# ----------------------------------------------------------------------------
LOG_FIELDS = [
    "session_id", "participant_id", "experimental_group", "dialogue_condition",
    "request_id", "npc_name", "location", "provider", "model",
    "bridge_received_utc", "model_started_utc", "model_finished_utc",
    "response_written_utc", "model_latency_ms", "player_input",
    "response_text", "quest_stage", "claimed_clues", "accepted_clues", "error",
]

def append_log(session_id: str, row: dict) -> None:
    path = LOGS_DIRECTORY / f"{safe_name(session_id)}_interactions.csv"
    file_has_content = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        if not file_has_content:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})
