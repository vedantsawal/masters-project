from pathlib import Path
from urllib.parse import urlparse
import csv
import hashlib
import ipaddress
import json


MP_ROOT = Path("/pscratch/sd/v/vinay/masters-project")

DATASET_ROOT = (
    MP_ROOT
    / "data/Malicious-PowerShell-Dataset"
)

CANDIDATE_POOL = (
    MP_ROOT
    / "step5/data/candidate_pool.csv"
)

SAMPLE_DECISIONS = (
    MP_ROOT
    / "step5/data/manual_sample_decisions.csv"
)

IOC_DECISIONS = (
    MP_ROOT
    / "step5/data/manual_ioc_decisions.csv"
)

OUT_MANIFEST = (
    MP_ROOT
    / "step5/data/pilot_manifest.csv"
)

OUT_GROUND_TRUTH = (
    MP_ROOT
    / "step5/data/ground_truth_iocs.jsonl"
)

REPRO_DIR = (
    MP_ROOT
    / "step5/repro"
)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def is_ip(value):
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def domain_from_url(url):
    try:
        host = urlparse(url).hostname

        if not host:
            return None

        host = host.lower().rstrip(".")

        # IP-hosted URLs should not create a duplicate
        # "domain" ground-truth item.
        if is_ip(host):
            return None

        return host

    except Exception:
        return None


# ============================================================
# Load candidate metadata
# ============================================================

with CANDIDATE_POOL.open(
    newline="",
    encoding="utf-8",
) as f:
    candidate_rows = list(csv.DictReader(f))

candidate_by_id = {
    row["candidate_id"]: row
    for row in candidate_rows
}


# ============================================================
# Load sample-level decisions
# ============================================================

with SAMPLE_DECISIONS.open(
    newline="",
    encoding="utf-8",
) as f:
    sample_rows = list(csv.DictReader(f))


accepted_ids = [
    row["candidate_id"]
    for row in sample_rows
    if row["accept_sample"].strip().upper() == "Y"
]


if len(accepted_ids) != 30:
    raise RuntimeError(
        f"Expected exactly 30 accepted samples, "
        f"found {len(accepted_ids)}"
    )


if len(set(accepted_ids)) != 30:
    raise RuntimeError(
        "Duplicate candidate IDs found among accepted samples"
    )


# Preserve candidate ordering.
accepted_ids = sorted(
    accepted_ids,
    key=lambda x: int(x[1:])
)


# ============================================================
# Load manually validated IOC decisions
# ============================================================

with IOC_DECISIONS.open(
    newline="",
    encoding="utf-8",
) as f:
    ioc_rows = list(csv.DictReader(f))


validated = {}

for row in ioc_rows:

    if row["keep"].strip().upper() != "Y":
        continue

    cid = row["candidate_id"]

    if cid not in accepted_ids:
        continue

    indicator_type = row["indicator_type"].strip()
    value = row["normalized_value"].strip()

    if not value:
        continue

    validated.setdefault(
        cid,
        {
            "url": set(),
            "ipv4": set(),
        },
    )

    if indicator_type not in {"url", "ipv4"}:
        continue

    validated[cid][indicator_type].add(value)


# ============================================================
# Build frozen records
# ============================================================

ground_truth_records = []
manifest_rows = []

seen_hashes = set()


for cid in accepted_ids:

    if cid not in candidate_by_id:
        raise RuntimeError(
            f"{cid} missing from candidate_pool.csv"
        )

    meta = candidate_by_id[cid]

    sample_path = (
        DATASET_ROOT
        / meta["relative_path"]
    )

    if not sample_path.exists():
        raise RuntimeError(
            f"Missing sample file: {sample_path}"
        )

    actual_sha256 = sha256_file(sample_path)

    filename = sample_path.name
    filename_stem = sample_path.stem

    # Dataset filenames are SHA-256 hashes.
    if (
        len(filename_stem) == 64
        and actual_sha256 != filename_stem
    ):
        raise RuntimeError(
            f"SHA-256 mismatch for {cid}: "
            f"{actual_sha256} != {filename_stem}"
        )

    if actual_sha256 in seen_hashes:
        raise RuntimeError(
            f"Exact duplicate file detected: {cid}"
        )

    seen_hashes.add(actual_sha256)


    decisions = validated.get(
        cid,
        {
            "url": set(),
            "ipv4": set(),
        },
    )

    urls = sorted(decisions["url"])
    ipv4s = sorted(decisions["ipv4"])

    # Derive domains only from manually validated URLs.
    domains = sorted({
        domain
        for url in urls
        if (domain := domain_from_url(url))
    })


    primary_count = (
        len(urls)
        + len(ipv4s)
    )

    if primary_count == 0:
        raise RuntimeError(
            f"{cid} is accepted but has no validated "
            f"URL or IPv4 ground truth"
        )


    record = {
        "candidate_id": cid,
        "sha256": actual_sha256,
        "filename": filename,
        "relative_path": meta["relative_path"],

        "iocs": {
            "urls": urls,
            "ipv4": ipv4s,
            "domains_from_urls": domains,
        },

        "counts": {
            "urls": len(urls),
            "ipv4": len(ipv4s),
            "domains_from_urls": len(domains),
            "primary_iocs": primary_count,
        },

        "ground_truth_policy": (
            "Primary ground truth consists of manually "
            "validated literal URLs and IPv4 addresses. "
            "Domains are deterministically derived from "
            "validated URL hostnames and are recorded "
            "separately to avoid treating them as "
            "independent manual annotations."
        ),
    }

    ground_truth_records.append(record)


    manifest_rows.append({
        "pilot_id": f"P{len(manifest_rows)+1:03d}",
        "candidate_id": cid,
        "sha256": actual_sha256,
        "filename": filename,
        "relative_path": meta["relative_path"],
        "bytes": meta["bytes"],
        "script_tokens": meta["script_tokens"],
        "gold_urls": len(urls),
        "gold_ipv4": len(ipv4s),
        "derived_domains": len(domains),
        "primary_gold_iocs": primary_count,
    })


# ============================================================
# Write ground_truth_iocs.jsonl
# ============================================================

with OUT_GROUND_TRUTH.open(
    "w",
    encoding="utf-8",
) as f:

    for record in ground_truth_records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=False,
            )
            + "\n"
        )


# ============================================================
# Write pilot_manifest.csv
# ============================================================

with OUT_MANIFEST.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fields = [
        "pilot_id",
        "candidate_id",
        "sha256",
        "filename",
        "relative_path",
        "bytes",
        "script_tokens",
        "gold_urls",
        "gold_ipv4",
        "derived_domains",
        "primary_gold_iocs",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(manifest_rows)


# ============================================================
# Freeze artifact hashes
# ============================================================

REPRO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

gt_hash = sha256_file(
    OUT_GROUND_TRUTH
)

manifest_hash = sha256_file(
    OUT_MANIFEST
)


with (
    REPRO_DIR
    / "ground_truth_iocs.sha256"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        f"{gt_hash}  "
        f"{OUT_GROUND_TRUTH.name}\n"
    )


with (
    REPRO_DIR
    / "pilot_manifest.sha256"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        f"{manifest_hash}  "
        f"{OUT_MANIFEST.name}\n"
    )


# ============================================================
# Summary
# ============================================================

total_urls = sum(
    r["counts"]["urls"]
    for r in ground_truth_records
)

total_ipv4 = sum(
    r["counts"]["ipv4"]
    for r in ground_truth_records
)

total_domains = sum(
    r["counts"]["domains_from_urls"]
    for r in ground_truth_records
)

total_primary = sum(
    r["counts"]["primary_iocs"]
    for r in ground_truth_records
)


print(
    "========== FROZEN STEP 5 PILOT =========="
)

print(
    "Accepted samples:",
    len(ground_truth_records),
)

print(
    "Unique sample SHA-256s:",
    len(seen_hashes),
)

print()
print("Primary ground truth:")
print("  URLs:", total_urls)
print("  IPv4:", total_ipv4)
print("  Total primary IOCs:", total_primary)

print()
print(
    "Derived URL hostnames:",
    total_domains,
)

print()
print("Ground truth:")
print(OUT_GROUND_TRUTH)

print()
print("Pilot manifest:")
print(OUT_MANIFEST)

print()
print("Ground-truth SHA-256:")
print(gt_hash)

print()
print("Manifest SHA-256:")
print(manifest_hash)

print(
    "========================================="
)
