from pathlib import Path
import argparse
import csv
import importlib.util
import json

from vllm import LLM, SamplingParams


MP_ROOT = Path("/pscratch/sd/v/vinay/masters-project")

ATTACKED_MANIFEST = (
    MP_ROOT
    / "step5/data/attacked_manifest.csv"
)

DEFAULT_OUTPUT = (
    MP_ROOT
    / "step5/results/attacked_vicuna_raw.jsonl"
)


# Load the already-frozen clean inference implementation.
clean_script = (
    MP_ROOT
    / "step5/scripts/run_clean_vicuna.py"
)

spec = importlib.util.spec_from_file_location(
    "clean_runner",
    clean_script,
)

clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


MODEL_NAME = clean.MODEL_NAME


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    with ATTACKED_MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if args.limit is not None:
        rows = rows[:args.limit]

    prompts = []
    metadata = []

    for row in rows:

        sample_path = (
            MP_ROOT
            / row["relative_path"]
        )

        if not sample_path.exists():
            raise FileNotFoundError(
                sample_path
            )

        script = clean.read_text(
            sample_path
        )

        # EXACT same prompt builder used in clean condition.
        prompts.append(
            clean.build_prompt(script)
        )

        metadata.append({
            "pilot_id":
                row["pilot_id"],

            "candidate_id":
                row["candidate_id"],

            "original_sha256":
                row["original_sha256"],

            "attacked_sha256":
                row["attacked_sha256"],

            "attack_name":
                row["attack_name"],

            "attack_location":
                row["attack_location"],
        })

    print("Model:", MODEL_NAME)
    print("Samples:", len(rows))
    print("Attack: naive_ioc_suppression")
    print("Attack location: end")
    print("Output:", args.output)

    print("Loading Vicuna...")

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )

    print("Running attacked inference...")

    outputs = llm.generate(
        prompts,
        sampling,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for meta, result in zip(
            metadata,
            outputs,
        ):

            raw_output = (
                result.outputs[0]
                .text
                .strip()
            )

            record = {
                **meta,
                "model": MODEL_NAME,
                "temperature": 0.0,
                "max_tokens": 256,
                "raw_output": raw_output,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        "========== ATTACKED INFERENCE COMPLETE =========="
    )
    print("Samples:", len(rows))
    print("Output:", args.output)
    print(
        "================================================="
    )


if __name__ == "__main__":
    main()
