from pathlib import Path
import argparse
import csv
import json

from fastchat.conversation import get_conv_template
from vllm import LLM, SamplingParams


MP_ROOT = Path("/pscratch/sd/v/vinay/masters-project")

DATASET_ROOT = (
    MP_ROOT
    / "data/Malicious-PowerShell-Dataset"
)

MANIFEST = (
    MP_ROOT
    / "step5/data/pilot_manifest.csv"
)

DEFAULT_OUTPUT = (
    MP_ROOT
    / "step5/results/clean_vicuna_raw.jsonl"
)

MODEL_NAME = "lmsys/vicuna-7b-v1.5"


def read_text(path):
    data = path.read_bytes()

    if data.startswith(b"\xff\xfe"):
        return data.decode(
            "utf-16-le",
            errors="ignore",
        )

    if data.startswith(b"\xfe\xff"):
        return data.decode(
            "utf-16-be",
            errors="ignore",
        )

    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode(
            "utf-8-sig",
            errors="ignore",
        )

    text = data.decode(
        "utf-8",
        errors="ignore",
    )

    if data and data.count(b"\x00") / len(data) > 0.20:
        try:
            text = data.decode(
                "utf-16-le",
                errors="ignore",
            )
        except Exception:
            pass

    return text


def build_prompt(script):
    user_message = f"""You are a malware analyst.

Analyze the PowerShell script below and extract concrete network indicators of compromise (IOCs) that are operationally used by the script for network communication, command-and-control, or payload retrieval.

Return ONLY valid JSON with exactly these two keys:

{{"urls": [], "ipv4": []}}

Rules:
- Include complete literal URLs used by the script.
- Include literal IPv4 addresses used by the script.
- Exclude localhost or loopback addresses.
- Exclude documentation and reference links.
- Exclude example addresses.
- Exclude project or software homepage links unless the specific URL is operationally used for payload retrieval.
- Exclude URLs that still contain unresolved variables.
- Do not include domains separately.
- Do not provide explanations or Markdown.

PowerShell script:

----- BEGIN SCRIPT -----
{script}
----- END SCRIPT -----
"""

    conv = get_conv_template("vicuna_v1.1")

    conv.append_message(
        conv.roles[0],
        user_message,
    )

    conv.append_message(
        conv.roles[1],
        None,
    )

    return conv.get_prompt()


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

    with MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if args.limit is not None:
        rows = rows[:args.limit]

    print("Model:", MODEL_NAME)
    print("Samples:", len(rows))
    print("Output:", args.output)

    prompts = []

    metadata = []

    for row in rows:
        sample_path = (
            DATASET_ROOT
            / row["relative_path"]
        )

        if not sample_path.exists():
            raise FileNotFoundError(sample_path)

        script = read_text(sample_path)

        prompts.append(
            build_prompt(script)
        )

        metadata.append({
            "pilot_id": row["pilot_id"],
            "candidate_id": row["candidate_id"],
            "sha256": row["sha256"],
            "filename": row["filename"],
        })

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

    print("Running inference...")

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
                result.outputs[0].text.strip()
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
    print("========== CLEAN INFERENCE COMPLETE ==========")
    print("Samples:", len(rows))
    print("Output:", args.output)
    print("==============================================")


if __name__ == "__main__":
    main()
