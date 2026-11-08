from pathlib import Path
import csv
import ipaddress
import random
import re

from transformers import AutoTokenizer


SEED = 2026

MIN_FILE_BYTES = 200
MAX_FILE_BYTES = 20_000

MAX_SCRIPT_TOKENS = 2500
POOL_SIZE = 60

ROOT = Path(
    "/pscratch/sd/v/vinay/masters-project/"
    "data/Malicious-PowerShell-Dataset/malicious_samples"
)

OUTPUT = Path(
    "/pscratch/sd/v/vinay/masters-project/"
    "step5/data/candidate_pool.csv"
)


# ============================================================
# Candidate IOC regexes
# ============================================================

URL_RE = re.compile(
    r"""https?://[^\s'"<>`]+""",
    re.IGNORECASE,
)

IP_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

DOMAIN_RE = re.compile(
    r"\b(?:[A-Za-z0-9]"
    r"(?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,24}\b"
)

WIN_PATH_RE = re.compile(
    r"""[A-Za-z]:\\[^\r\n"'<>|]+"""
)


# ============================================================
# Decode script as TEXT ONLY
# ============================================================

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

    # UTF-16LE heuristic for files without BOM.
    if data and data.count(b"\x00") / len(data) > 0.20:
        try:
            text = data.decode(
                "utf-16-le",
                errors="ignore",
            )
        except Exception:
            pass

    return text


# ============================================================
# IOC candidate extraction
# ============================================================

def valid_ips(text):
    result = set()

    for candidate in IP_RE.findall(text):

        try:
            ip = ipaddress.ip_address(candidate)

            if ip.version == 4:
                result.add(candidate)

        except ValueError:
            pass

    return result


def extract_candidates(text):

    urls = set(
        URL_RE.findall(text)
    )

    ips = valid_ips(text)

    domains = set(
        DOMAIN_RE.findall(text)
    )

    paths = set(
        WIN_PATH_RE.findall(text)
    )

    # Strong candidates are substantially less ambiguous
    # than arbitrary dotted strings / bare domains.
    strong_total = (
        len(urls)
        + len(ips)
        + len(paths)
    )

    return {
        "url_count": len(urls),
        "ip_count": len(ips),
        "domain_count": len(domains),
        "path_count": len(paths),
        "strong_candidate_total": strong_total,
        "candidate_total": (
            strong_total
            + len(domains)
        ),
    }


# ============================================================
# Load tokenizer
# ============================================================

print("Loading Vicuna tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "lmsys/vicuna-7b-v1.5",
    use_fast=False,
    local_files_only=True,
)

# We are only COUNTING tokens here.
# Actual model inputs will still use the real 4096-token limit.
tokenizer.model_max_length = 1_000_000

print("Tokenizer loaded.")


# ============================================================
# Build eligible population
# ============================================================

rows = []

files_seen = 0
excluded_obf = 0
excluded_size = 0
excluded_no_strong_ioc = 0
excluded_tokens = 0


for path in sorted(ROOT.glob("*.ps1")):

    files_seen += 1

    # Exclude repository-created obfuscated derivatives.
    if "-obf-out-" in path.name:
        excluded_obf += 1
        continue

    size = path.stat().st_size

    if not (
        MIN_FILE_BYTES
        <= size
        <= MAX_FILE_BYTES
    ):
        excluded_size += 1
        continue

    text = read_text(path)

    counts = extract_candidates(text)

    # Do not qualify samples only because a dotted string
    # looked like a domain.
    if counts["strong_candidate_total"] == 0:
        excluded_no_strong_ioc += 1
        continue

    token_count = len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )

    if token_count > MAX_SCRIPT_TOKENS:
        excluded_tokens += 1
        continue

    rows.append({
        "filename": path.name,
        "relative_path":
            f"malicious_samples/{path.name}",
        "bytes": size,
        "script_tokens": token_count,
        **counts,
    })


print()
print(
    "Eligible before sampling:",
    len(rows),
)


# ============================================================
# Deterministic sampling
# ============================================================

rng = random.Random(SEED)

rng.shuffle(rows)

selected = rows[:POOL_SIZE]


# ============================================================
# Save manifest
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "candidate_id",
            "filename",
            "relative_path",
            "bytes",
            "script_tokens",
            "url_count",
            "ip_count",
            "domain_count",
            "path_count",
            "strong_candidate_total",
            "candidate_total",
        ],
    )

    writer.writeheader()

    for i, row in enumerate(
        selected,
        start=1,
    ):

        writer.writerow({
            "candidate_id":
                f"C{i:03d}",
            **row,
        })


# ============================================================
# Summary
# ============================================================

print()
print(
    "========== CANDIDATE POOL V2 =========="
)

print(
    "Files seen:",
    files_seen,
)

print(
    "Excluded derived obfuscation:",
    excluded_obf,
)

print(
    "Excluded by size:",
    excluded_size,
)

print(
    "Excluded without strong IOC:",
    excluded_no_strong_ioc,
)

print(
    "Excluded by token length:",
    excluded_tokens,
)

print(
    "Eligible samples:",
    len(rows),
)

print(
    "Candidate pool selected:",
    len(selected),
)

print(
    "Seed:",
    SEED,
)

print(
    "Max script tokens:",
    MAX_SCRIPT_TOKENS,
)

print()

print("Saved:")
print(OUTPUT)

print(
    "======================================="
)
