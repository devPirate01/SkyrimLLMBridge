"""
llm_api_client.py

Purpose:
    Exclusively handles HTTP requests to KoboldCpp and Open WebUI.
"""

import json
import time
from pathlib import Path
import requests

BRIDGE_DIRECTORY = Path(__file__).resolve().parent
MODELS_DIR = BRIDGE_DIRECTORY / "model_parameters"
DEFAULT_PARAMS_PATH = MODELS_DIR / "default_model_parameters.json"

# Connection pooling for massive latency reduction
session = requests.Session()

class LLMAPIError(RuntimeError):
    """Raised when the LLM provider fails."""

def provider_settings(
    provider: str,
    koboldcpp_url: str,
    koboldcpp_model: str,
    openwebui_url: str,
    openwebui_token: str,
    openwebui_model: str,
) -> tuple[str, str, dict]:
    """Return endpoint, model ID, and request headers for the selected provider."""
    provider = provider.strip().lower()
    if provider == "koboldcpp":
        return (
            f"{koboldcpp_url.rstrip('/')}/chat/completions",
            koboldcpp_model,
            {"Content-Type": "application/json"},
        )
    elif provider == "openwebui":
        if not openwebui_url or not openwebui_token:
            raise LLMAPIError("OPENWEBUI_URL and OPENWEBUI_TOKEN are required.")
        return (
            f"{openwebui_url.rstrip('/')}/api/chat/completions",
            openwebui_model,
            {
                "Authorization": f"Bearer {openwebui_token}",
                "Content-Type": "application/json",
            },
        )
    raise LLMAPIError("LLM_PROVIDER must be 'koboldcpp' or 'openwebui'.")

def load_model_parameters(model: str) -> dict:
    """Load model-specific parameters or fallback to default."""
    # Sanitize model name for filename (e.g. llama3.3:latest -> llama3.3)
    safe_model = model.split(":")[0].replace("/", "_")
    model_params_path = MODELS_DIR / f"{safe_model}_parameters.json"

    if model_params_path.exists():
        path_to_load = model_params_path
    elif DEFAULT_PARAMS_PATH.exists():
        path_to_load = DEFAULT_PARAMS_PATH
    else:
        return {"temperature": 0.2, "max_tokens": 350, "top_k": 40, "top_p": 0.9}

    try:
        return json.loads(path_to_load.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"Warning: Failed to read {path_to_load.name}: {error}")
        return {"temperature": 0.2, "max_tokens": 350, "top_k": 40, "top_p": 0.9}

def call_llm(
    provider: str,
    url: str,
    model: str,
    headers: dict,
    messages: list[dict],
    parameters: dict,
    timeout_seconds: int = 180,
) -> tuple[str, float]:
    """Call the API and return the raw text output and latency."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": parameters.get("temperature", 0.2),
        "max_tokens": parameters.get("max_tokens", 350),
        "top_k": parameters.get("top_k", 40),
        "top_p": parameters.get("top_p", 0.9),
    }

    started_counter = time.perf_counter()
    try:
        response = session.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
    except Exception as e:
        raise LLMAPIError(f"Network error: {e}") from e

    latency = time.perf_counter() - started_counter

    if response.status_code != 200:
        raise LLMAPIError(
            f"{provider} request failed: {response.status_code} {response.text}"
        )

    try:
        response_data = response.json()
        raw_output = response_data["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise LLMAPIError(f"Unexpected {provider} response format: {response.text}") from error

    if not raw_output:
        raise LLMAPIError(f"{provider} returned an empty response.")

    return raw_output, latency
