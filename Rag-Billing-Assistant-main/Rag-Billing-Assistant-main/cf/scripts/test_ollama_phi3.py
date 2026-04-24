import json
import os
import sys
import urllib.error
import urllib.request


def call_ollama(prompt: str, model: str = "phi3:mini") -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    endpoint = f"{base_url.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed.get("response", "").strip()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Make sure Ollama is running on http://localhost:11434 "
            "and the model is pulled (ollama pull phi3:mini)."
        ) from exc


def main() -> int:
    prompt = "Explain SaaS billing in 10 words"
    model = os.getenv("OLLAMA_MODEL", "phi3:mini")

    print(f"Testing Ollama endpoint with model: {model}")
    print(f"Prompt: {prompt}")

    try:
        answer = call_ollama(prompt=prompt, model=model)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    print("\nSUCCESS: Ollama is accessible.")
    print(f"Response: {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
