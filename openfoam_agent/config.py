from __future__ import annotations

from pathlib import Path
from typing import Optional

import os as _os

_OF_ROOT = _os.environ.get(
    "OPENFOAM_ROOT", "/usr/lib/openfoam/openfoam2412"
)
OPENFOAM_BASHRC = _os.environ.get(
    "OPENFOAM_BASHRC", f"{_OF_ROOT}/etc/bashrc"
)
TUTORIALS_DIR = Path(
    _os.environ.get("OPENFOAM_TUTORIALS", f"{_OF_ROOT}/tutorials")
)
PROJECT_ROOT = Path(__file__).parent.parent

CASES_DIR = PROJECT_ROOT / "data/cases"
LOGS_DIR = PROJECT_ROOT / "data/logs"
DATASET_DIR = PROJECT_ROOT / "data/dataset"
CHECKPOINTS_DIR = PROJECT_ROOT / "data/checkpoints"
CHROMA_DIR = PROJECT_ROOT / "data/chroma_db"

LLM_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
# Layer-3 evolve.sh sets this to point the current subprocess at a candidate
# adapter for the regression-gate eval, without touching the production
# config.py value.
LLM_MODEL = _os.environ.get("OPENFOAM_AGENT_LLM_OVERRIDE", LLM_MODEL)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SEQ_LEN = 4096
VLLM_GPU_MEMORY_FRACTION = float(_os.environ.get("VLLM_GPU_MEM_FRAC", "0.90"))
VLLM_MAX_NUM_SEQS = int(_os.environ.get("VLLM_MAX_NUM_SEQS", "32"))

# ── Self-evolution knobs (Layer 1 / 2 / 3) ──────────────────────────────────
# Score below this triggers a Layer-1 self-correction retry. Set ABOVE the
# 0.5 capture threshold so that any borderline run (didn't crash, but isn't
# clean either) still gets a retry attempt with enriched failure context.
# This is also what feeds Layer-4 DPO: the (low first attempt, higher retry)
# pair on the same prompt is exactly the preference signal DPO needs.
# Was 0.4 originally — moved to 0.7 once we learned that anything 0.5–0.7
# almost never triggered retry in practice, starving DPO of pairs.
RETRY_SCORE_THRESHOLD = float(_os.environ.get("RETRY_SCORE_THRESHOLD", "0.7"))
# Minimum score for a row to enter the curated training corpus (Layer 2).
# Higher than the 0.5 capture threshold so we only train on *good* runs.
MIN_RETRAIN_SCORE = float(_os.environ.get("MIN_RETRAIN_SCORE", "0.65"))
# Fire evolve.sh once dataset.json has accumulated this many new high-score
# rows since the last successful evolution cycle (Layer 3).
EVOLUTION_BATCH_SIZE = int(_os.environ.get("EVOLUTION_BATCH_SIZE", "25"))
# Pinned baseline for the regression gate. evolve.sh refuses to swap if a
# fresh adapter regresses below these numbers on data/eval/ood_100_v2.json.
EVAL_GATE_FILE = PROJECT_ROOT / "data/eval/regression_gate.json"
# Where every retry attempt (success or fail) is appended for retry-pair
# mining. Distinct from dataset.json (curated, score≥0.5 only).
ATTEMPTS_LOG = DATASET_DIR / "attempts.jsonl"

# ── Anti-collapse defenses (Layers 5/6/7) ───────────────────────────────────
# Frozen v1 corpus (the original 402 rows the 14 B model was first trained on).
# Mixed back into every evolve cycle to prevent forgetting. NEVER overwrite
# this file — curate_dataset.py writes to its own versioned output by default.
ANCHOR_DATASET = DATASET_DIR / "anchor_v1_402.jsonl"
# Fraction of the training mix that should come from the anchor on each cycle.
# 0.0 disables; 0.3 means ~30 % anchor + 70 % new high-score rows.
EVOLUTION_ANCHOR_FRACTION = float(_os.environ.get("EVOLUTION_ANCHOR_FRACTION", "0.3"))
# When 1, evolve.sh runs active_learning.py to author + score new prompts in
# the weakest solver family from the previous eval. Skipped automatically if
# the previous eval had no family below the active-learning threshold.
EVOLUTION_ACTIVE_LEARNING = int(_os.environ.get("EVOLUTION_ACTIVE_LEARNING", "1"))
# Match-rate below this triggers active-learning targeting for that family.
ACTIVE_LEARNING_THRESHOLD = float(_os.environ.get("ACTIVE_LEARNING_THRESHOLD", "0.95"))

_llm_instance = None

USE_CPU_INFERENCE = _os.environ.get("USE_CPU_INFERENCE", "0") == "1"


class _TransformersOutput:
    def __init__(self, text: str):
        self.text = text


class _TransformersRequestOutput:
    def __init__(self, text: str):
        self.outputs = [_TransformersOutput(text)]


class TransformersLLM:
    """CPU-compatible drop-in for vllm.LLM — uses HuggingFace transformers."""

    def __init__(self, model: str, max_model_len: int = 4096, **_kwargs):
        # Patch torch._dynamo dispatch table to be idempotent BEFORE importing
        # transformers.  Without this, reloading config or certain import orders
        # raise "Duplicate dispatch rule for <built-in function intern>".
        try:
            import torch._dynamo.variables.builtin as _vb
            if hasattr(_vb, "VariableBuilder") and hasattr(_vb.VariableBuilder, "id_dispatch"):
                _orig = _vb.VariableBuilder.id_dispatch

                class _IdempotentDict(type(_orig)):
                    def __setitem__(self, k, v):
                        if k not in self:
                            super().__setitem__(k, v)

                _vb.VariableBuilder.id_dispatch = _IdempotentDict(_orig)
        except Exception:
            pass

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
        )
        self._max_model_len = max_model_len

    def chat(self, messages: list, sampling_params=None):
        import torch
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self._max_model_len - 512,
        )
        max_new = getattr(sampling_params, "max_tokens", 512)
        temp = float(getattr(sampling_params, "temperature", 0.7))
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=max(temp, 0.01),
                do_sample=temp > 0.01,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        result = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return [_TransformersRequestOutput(result)]


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        model = _os.environ.get("OPENFOAM_AGENT_LLM_OVERRIDE", LLM_MODEL)
        # Read at call-time so GUI device toggle works without reload
        use_cpu = _os.environ.get("USE_CPU_INFERENCE", "0") == "1"
        if use_cpu:
            _llm_instance = TransformersLLM(model=model, max_model_len=MAX_SEQ_LEN)
        else:
            from vllm import LLM
            _llm_instance = LLM(
                model=model,
                max_model_len=MAX_SEQ_LEN,
                gpu_memory_utilization=VLLM_GPU_MEMORY_FRACTION,
                max_num_seqs=VLLM_MAX_NUM_SEQS,
                dtype="float16",
            )
    return _llm_instance


def get_unsloth_model(load_for_training: bool = False):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LLM_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    if load_for_training:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    return model, tokenizer
