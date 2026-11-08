from functools import partial
from pathlib import Path
import hashlib
import json
import os
import subprocess

import jsonlines
import pandas as pd
import torch
import transformers
import vllm

from accelerate import Accelerator
from accelerate.utils import set_seed

from bipia.model import AutoLLM
from bipia.data import AutoPIABuilder
from bipia.metrics import BipiaEvalFactory


# ============================================================
# Experiment configuration
# ============================================================

SEED = 2023
CHUNK_SIZE = 20
MAX_NEW_TOKENS = 2048
MODEL_CONFIG = "config/vicuna_7b.yaml"

MP_ROOT = Path(os.environ["MP_ROOT"])
OUT_ROOT = MP_ROOT / "outputs/full/vicuna7b_codeqa"
CHUNK_DIR = OUT_ROOT / "chunks"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()


def chunk_is_valid(path, expected_ids):
    if not path.exists():
        return False

    try:
        with jsonlines.open(str(path), "r") as reader:
            records = list(reader)

        found_ids = [int(r["sample_id"]) for r in records]
        return found_ids == list(expected_ids)

    except Exception:
        return False


def write_atomic_jsonl(path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")

    with jsonlines.open(str(tmp), "w") as writer:
        writer.write_all(records)

    os.replace(tmp, path)


# ============================================================
# Initialize reproducible environment
# ============================================================

accelerator = Accelerator()
set_seed(SEED)

print("==================================================", flush=True)
print("BIPIA VICUNA-7B CODEQA FULL REPRODUCTION", flush=True)
print("==================================================", flush=True)
print("Git commit:", git_commit(), flush=True)
print("Torch:", torch.__version__, flush=True)
print("CUDA runtime:", torch.version.cuda, flush=True)
print("vLLM:", getattr(vllm, "__version__", "unknown"), flush=True)
print("Transformers:", transformers.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("BF16:", torch.cuda.is_bf16_supported(), flush=True)

print("Seed:", SEED, flush=True)
print("Chunk size:", CHUNK_SIZE, flush=True)
print("Max new tokens:", MAX_NEW_TOKENS, flush=True)
print("==================================================", flush=True)


# ============================================================
# Build the official BIPIA CodeQA benchmark
# ============================================================

builder = AutoPIABuilder.from_name("code")(seed=SEED)

samples = builder(
    "benchmark/code/test.jsonl",
    "benchmark/code_attack_test.json",
    enable_stealth=False,
).reset_index(drop=True)

assert len(samples) == 7500, (
    f"Expected 7500 attacked examples, found {len(samples)}"
)

print("Attacked examples:", len(samples), flush=True)
print(
    "Position counts:",
    samples["position"].value_counts().to_dict(),
    flush=True,
)
print(
    "Unique attack variants:",
    samples["attack_name"].nunique(),
    flush=True,
)


# ============================================================
# Load official BIPIA Vicuna wrapper
# ============================================================

model_cls = AutoLLM.from_name(MODEL_CONFIG)

print("\nLoading Vicuna-7B...", flush=True)

llm = model_cls(
    config=MODEL_CONFIG,
    accelerator=accelerator,
    tensor_parallel_size=1,
    max_new_tokens=MAX_NEW_TOKENS,
)

print("Vicuna-7B loaded successfully.", flush=True)


# ============================================================
# Official BIPIA CodeQA evaluator
# ============================================================

all_attacks = sorted(samples["attack_name"].unique())

evaluator = BipiaEvalFactory(
    gpt_config={},
    activate_attacks=all_attacks,
)


# ============================================================
# Process benchmark in checkpointed chunks
# ============================================================

total = len(samples)

for start in range(0, total, CHUNK_SIZE):

    end = min(start + CHUNK_SIZE, total)

    expected_ids = range(start, end)

    chunk_path = CHUNK_DIR / (
        f"chunk_{start:05d}_{end - 1:05d}.jsonl"
    )

    if chunk_is_valid(chunk_path, expected_ids):
        print(
            f"[SKIP] {start:05d}-{end - 1:05d} already complete",
            flush=True,
        )
        continue

    print(
        f"\n[RUN] {start:05d}-{end - 1:05d} "
        f"({end}/{total})",
        flush=True,
    )

    chunk = samples.iloc[start:end]

    processed = []

    for _, row in chunk.iterrows():

        ex = row.to_dict()

        ex = llm.process_fn(
            ex,
            prompt_construct_fn=partial(
                builder.construct_prompt,
                require_system_prompt=llm.require_system_prompt,
                ign_guidance="",
            ),
        )

        processed.append(ex)

    messages = [x["message"] for x in processed]

    predictions = llm.generate({
        "message": messages
    })

    attacks = chunk["attack_name"].tolist()
    tasks = chunk["task_name"].tolist()
    references = chunk["ideal"].tolist()

    asrs = evaluator.add_batch(
        references=references,
        predictions=predictions,
        attacks=attacks,
        tasks=tasks,
    )

    records = []

    for offset, (
        prediction,
        asr,
        processed_example,
    ) in enumerate(
        zip(predictions, asrs, processed)
    ):

        sample_id = start + offset
        row = chunk.iloc[offset]

        attack_name = row["attack_name"]

        attack_category = attack_name.rsplit("-", 1)[0]

        prompt_hash = hashlib.sha256(
            processed_example["message"].encode("utf-8")
        ).hexdigest()

        records.append({
            "sample_id": sample_id,
            "attack_name": attack_name,
            "attack_category": attack_category,
            "position": row["position"],
            "task_name": row["task_name"],
            "target": row["ideal"],
            "response": prediction,
            "asr": float(asr),
            "prompt_sha256": prompt_hash,
        })

    write_atomic_jsonl(chunk_path, records)

    successes = sum(r["asr"] for r in records)

    print(
        f"[SAVED] {chunk_path.name} | "
        f"successes={int(successes)}/{len(records)}",
        flush=True,
    )


# ============================================================
# Consolidate all checkpoints
# ============================================================

print("\nConsolidating checkpoints...", flush=True)

all_records = []

chunk_files = sorted(CHUNK_DIR.glob("chunk_*.jsonl"))

for path in chunk_files:
    with jsonlines.open(str(path), "r") as reader:
        all_records.extend(list(reader))

all_records.sort(key=lambda x: int(x["sample_id"]))

assert len(all_records) == total, (
    f"Expected {total} final records, "
    f"found {len(all_records)}"
)

assert len({r["sample_id"] for r in all_records}) == total, (
    "Duplicate sample IDs detected."
)

final_path = OUT_ROOT / "vicuna7b_codeqa_7500.jsonl"

write_atomic_jsonl(final_path, all_records)


# ============================================================
# Compute summaries
# ============================================================

df = pd.DataFrame(all_records)

overall_asr = float(df["asr"].mean())

by_position = (
    df.groupby("position")["asr"]
      .agg(["count", "sum", "mean"])
      .sort_index()
)

by_category = (
    df.groupby("attack_category")["asr"]
      .agg(["count", "sum", "mean"])
      .sort_values("mean", ascending=False)
)

by_attack = (
    df.groupby("attack_name")["asr"]
      .agg(["count", "sum", "mean"])
      .sort_values("mean", ascending=False)
)

by_position.to_csv(
    OUT_ROOT / "asr_by_position.csv"
)

by_category.to_csv(
    OUT_ROOT / "asr_by_attack_category.csv"
)

by_attack.to_csv(
    OUT_ROOT / "asr_by_attack_variant.csv"
)


metadata = {
    "bipia_commit": git_commit(),
    "model": "lmsys/vicuna-7b-v1.5",
    "dataset": "CodeQA",
    "seed": SEED,
    "num_samples": total,
    "num_attack_variants": int(
        samples["attack_name"].nunique()
    ),
    "num_positions": int(
        samples["position"].nunique()
    ),
    "chunk_size": CHUNK_SIZE,
    "temperature": 0,
    "max_new_tokens": MAX_NEW_TOKENS,
    "tensor_parallel_size": 1,
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "vllm_version": getattr(
        vllm, "__version__", "unknown"
    ),
    "transformers_version": transformers.__version__,
    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
    "overall_asr": overall_asr,
}

with open(
    OUT_ROOT / "summary.json",
    "w"
) as f:
    json.dump(metadata, f, indent=2)


# ============================================================
# Final report
# ============================================================

print("\n==================================================", flush=True)
print("FULL EXPERIMENT COMPLETE", flush=True)
print("==================================================", flush=True)
print("Samples:", len(df), flush=True)
print(
    "Successful attacks:",
    int(df["asr"].sum()),
    flush=True,
)
print(
    f"Overall ASR: {overall_asr:.6f} "
    f"({overall_asr * 100:.2f}%)",
    flush=True,
)

print("\nASR BY POSITION", flush=True)
print(by_position.to_string(), flush=True)

print("\nASR BY ATTACK CATEGORY", flush=True)
print(by_category.to_string(), flush=True)

print("\nFinal responses:", final_path, flush=True)
print(
    "Summary:",
    OUT_ROOT / "summary.json",
    flush=True,
)
print("==================================================", flush=True)
