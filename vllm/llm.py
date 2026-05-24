"""Lightweight transformers + bitsandbytes drop-in for vLLM's LLM class."""
from __future__ import annotations

import json
import re


class _RequestOutput:
    def __init__(self, text: str):
        self.text = text


class _CompletionOutput:
    def __init__(self, text: str):
        self.outputs = [_RequestOutput(text)]


class LLM:
    def __init__(
        self,
        model: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 32,
        dtype: str = "float16",
        **kwargs,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[vllm-stub] Loading {model} in 4-bit NF4 (bitsandbytes) ...")
        # Enable CPU offload so layers that don't fit in VRAM spill to RAM
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # Use gpu_memory_utilization to limit VRAM fraction; let the rest go to CPU
        max_mem = {
            0: f"{int(gpu_memory_utilization * torch.cuda.get_device_properties(0).total_memory / 1024**3 * 1024)}MiB",
            "cpu": "16GiB",
        } if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=True,
        )
        self.max_model_len = max_model_len
        print(f"[vllm-stub] Model ready. Device map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'N/A'}")

    def chat(self, messages: list[dict], sampling_params=None) -> list[_CompletionOutput]:
        import torch

        sp = sampling_params
        temperature = getattr(sp, "temperature", 0.7) if sp else 0.7
        max_tokens = getattr(sp, "max_tokens", 512) if sp else 512
        structured = getattr(sp, "structured_outputs", None) if sp else None

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(float(temperature), 1e-6),
                do_sample=float(temperature) > 0.01,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        result_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if structured and getattr(structured, "json", None) is not None:
            print(f"[vllm-stub] raw output: {result_text[:200]!r}")
            result_text = self._extract_json(result_text)

        return [_CompletionOutput(result_text)]

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        # Strip markdown code fences
        for fence in ("```json", "```"):
            if fence in text:
                text = text.split(fence, 1)[-1]
                text = text.rsplit("```", 1)[0].strip()
                break
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        # Try to pull out the outermost {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        return text
