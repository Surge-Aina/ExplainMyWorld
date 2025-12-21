import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "explain_world.txt"

# CPU-friendly instruct model
# LLM_ID = "Qwen/Qwen2.5-3B-Instruct"
LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
_model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
)

def _system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")

def _build_user_prompt(image_caption: str, user_text: str) -> str:
    user_text = (user_text or "").strip()
    return f"""IMAGE_CAPTION:
{image_caption}

USER_CONTEXT:
{user_text if user_text else "(none)"}

Return ONLY valid JSON with the required keys."""

def _extract_json(text: str) -> dict:
    """
    Robust-ish JSON extraction: finds first {...} block and parses it.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end+1])

def explain_world(image_caption: str, user_context: str) -> dict:
    system = _system_prompt()
    user = _build_user_prompt(image_caption, user_context)

    # Chat template (Qwen supports chat style)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = _tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            temperature=0.2,
        )

    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)

    try:
        return _extract_json(decoded)
    except Exception:
        # fallback: ask again in a stricter way (one retry)
        retry_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user + "\n\nIMPORTANT: Output ONLY JSON. No extra text."},
        ]
        retry_prompt = _tokenizer.apply_chat_template(retry_messages, tokenize=False, add_generation_prompt=True)
        retry_inputs = _tokenizer(retry_prompt, return_tensors="pt")

        with torch.no_grad():
            retry_out = _model.generate(
                **retry_inputs,
                max_new_tokens=350,
                do_sample=False,
                temperature=0.2,
            )
        retry_decoded = _tokenizer.decode(retry_out[0], skip_special_tokens=True)

        try:
            return _extract_json(retry_decoded)
        except Exception:
            return {
                "observed": [image_caption],
                "likely_causes": [],
                "why": "I couldn't format the reasoning as JSON, but the caption describes the scene above.",
                "confidence": "low",
                "question": "What time of day and where was this taken?"
            }
