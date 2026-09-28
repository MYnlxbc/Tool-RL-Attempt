#!/usr/bin/env python
"""Minimal Qwen3.5-4B smoke test for text-only tool-agent experiments.

The script intentionally does not execute a real tool.  It checks the pieces
needed before implementing rollout/RL:

* loading the Qwen3.5 checkpoint with the currently installed Transformers;
* applying the model chat template with thinking disabled by default;
* generating one response and extracting token-level transition log-probs;
* detecting a tool-call-looking JSON payload in the response.

Qwen3.5 is a multimodal checkpoint.  AppWorld is text-only, so this script
passes text messages only and leaves image/video inputs unused.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "You have one tool named calculator. Return exactly one JSON object "
            "for the next action. Calculate 123 * 456."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3.5 thinking. Default: disabled for tool-call smoke tests.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    return parser.parse_args()


def import_transformers() -> tuple[Any, Any, Any]:
    """Import Transformers lazily so --help works without ML dependencies."""

    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install a recent Transformers build and PyTorch "
            "before running this script. Qwen3.5 currently requires a recent "
            "Transformers version (often the main branch)."
        ) from exc
    return torch, transformers, transformers


def choose_model_loader(transformers: Any) -> Any:
    """Prefer the multimodal auto class, with a causal-LM fallback.

    The exact auto class can vary with the installed Transformers version. The
    fallback is useful for text-only converted checkpoints and older APIs.
    """

    loader = getattr(transformers, "AutoModelForImageTextToText", None)
    if loader is not None:
        return loader
    loader = getattr(transformers, "AutoModelForVision2Seq", None)
    if loader is not None:
        return loader
    loader = getattr(transformers, "AutoModelForCausalLM", None)
    if loader is not None:
        return loader
    raise RuntimeError("No compatible Transformers auto model class was found.")


def load_model_and_processor(model_id: str, torch: Any, transformers: Any):
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
    except ImportError as exc:
        message = str(exc).lower()
        if "pil" in message or "pillow" in message or "image processor" in message:
            raise RuntimeError(
                "Qwen3.5's processor needs Pillow for its vision-capable "
                "checkpoint, even when the input is text-only. Install it with "
                "`python -m pip install -U pillow`. The model card also "
                "recommends torchvision: `python -m pip install -U torchvision`."
            ) from exc
        raise

    loader = choose_model_loader(transformers)
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = loader.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model, processor, loader.__name__, dtype


def build_inputs(
    processor: Any,
    prompt: str,
    enable_thinking: bool,
    torch: Any,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool-using agent. Emit either a tool call or a final "
                "answer. Keep the action machine-readable."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Qwen3.5 exposes enable_thinking through chat_template_kwargs. Some
    # older processors do not accept that keyword, so provide a clear fallback
    # error rather than silently training with a different template.
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
    except TypeError as exc:
        raise RuntimeError(
            "The installed processor does not accept chat_template_kwargs. "
            "Install a current Transformers version compatible with Qwen3.5."
        ) from exc

    # Move every tensor (input_ids, attention_mask, and any processor metadata)
    # to the model's input device. Do not assume CUDA: CPU smoke tests are valid.
    device = next(iter(inputs.values())).device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def generate_with_logprobs(
    model: Any,
    inputs: dict[str, Any],
    tokenizer: Any,
    torch: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
) -> GenerationResult:
    do_sample = not greedy
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
    )
    if do_sample:
        generation_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.inference_mode():
        output = model.generate(**generation_kwargs)

    sequences = output.sequences
    input_ids = inputs["input_ids"]
    generated_ids = sequences[:, input_ids.shape[-1] :]

    # Transformers returns one score tensor per generated step. Transition
    # scores are normalized log-probabilities for the selected token.
    transition_scores = model.compute_transition_scores(
        sequences,
        output.scores,
        normalize_logits=True,
    )
    token_logprobs = transition_scores[0].detach().float().cpu().tolist()
    token_ids = generated_ids[0].detach().cpu().tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return GenerationResult(
        text=text,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
    )


def detect_tool_call(text: str) -> dict[str, Any] | None:
    """Best-effort extraction for smoke testing, not a production parser."""

    candidates = [text.strip()]
    for start_marker, end_marker in (("<tool_call>", "</tool_call>"), ("```json", "```")):
        start = text.find(start_marker)
        if start >= 0:
            start += len(start_marker)
            end = text.find(end_marker, start)
            candidates.append(text[start:end if end >= 0 else len(text)].strip())

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (
            value.get("type") in {"tool_call", "function"}
            or "tool_name" in value
            or "name" in value
        ):
            return value
    return None


def main() -> int:
    args = parse_args()

    try:
        torch, transformers, _ = import_transformers()
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        print(f"Transformers: {transformers.__version__}")
        print(f"Torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        model, processor, loader_name, dtype = load_model_and_processor(
            args.model,
            torch,
            transformers,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"Model loader: {loader_name}")
        print(f"Model dtype: {dtype}")
        print(f"Model parameters: {parameter_count / 1e9:.3f}B")
        print(f"Thinking enabled: {args.enable_thinking}")

        inputs = build_inputs(
            processor,
            args.prompt,
            args.enable_thinking,
            torch,
        )
        input_length = int(inputs["input_ids"].shape[-1])
        print(f"Input tokens: {input_length}")

        tokenizer = getattr(processor, "tokenizer", processor)
        result = generate_with_logprobs(
            model,
            inputs,
            tokenizer,
            torch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
        )

        print("\n--- generated text ---")
        print(result.text)
        print("--- token/logprob preview ---")
        for token_id, logprob in list(zip(result.token_ids, result.token_logprobs))[:20]:
            token = tokenizer.decode([token_id], skip_special_tokens=False)
            print(f"id={token_id:>7} logprob={logprob:>9.4f} token={token!r}")

        tool_call = detect_tool_call(result.text)
        print("--- smoke-test checks ---")
        print(f"Generated tokens: {len(result.token_ids)}")
        print(f"Finite logprobs: {all(torch.isfinite(torch.tensor(result.token_logprobs)).tolist())}")
        print(f"Tool-call-looking JSON detected: {tool_call is not None}")
        if tool_call is not None:
            print("Parsed candidate:")
            print(json.dumps(tool_call, ensure_ascii=False, indent=2))

        print("\nSmoke test completed. This script does not execute external tools.")
        return 0
    except Exception as exc:  # Keep failures readable for environment setup.
        print(f"SMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
