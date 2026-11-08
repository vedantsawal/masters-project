from pathlib import Path
from urllib.parse import urlparse
import csv
import hashlib
import ipaddress
import re


MP_ROOT = Path("/pscratch/sd/v/vinay/masters-project")

DATASET_ROOT = (
    MP_ROOT
    / "data/Malicious-PowerShell-Dataset"
)

SAMPLE_ROOT = DATASET_ROOT / "malicious_samples"

POOL = (
    MP_ROOT
    / "step5/data/candidate_pool.csv"
)

OUT_ROWS = (
    MP_ROOT
    / "step5/data/ioc_review.csv"
)

OUT_SUMMARY = (
    MP_ROOT
    / "step5/data/ioc_review_summary.csv"
)


# ============================================================
# Candidate patterns
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

PATH_RE = re.compile(
    r"""[A-Za-z]:\\[^\r\n"'<>|]+"""
)


# ============================================================
# Safe static decoding
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

    if data and data.count(b"\x00") / len(data) > 0.20:
        try:
            text = data.decode(
                "utf-16-le",
                errors="ignore",
            )
        except Exception:
            pass

    return text


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def valid_ipv4(candidate):
    try:
        ip = ipaddress.ip_address(candidate)
        return (
            candidate
            if ip.version == 4
            else None
        )
    except ValueError:
        return None


def clean_value(value):
    # Remove punctuation commonly captured after URLs/paths.
    return value.rstrip(
        ".,;:)]}"
    ).strip()


def context_snippet(text, start, end, radius=80):
    left = max(0, start - radius)
    right = min(len(text), end + radius)

    snippet = text[left:right]

    # Make CSV safe/easy to read.
    snippet = (
        snippet
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )

    snippet = re.sub(
        r"\s+",
        " ",
        snippet,
    )

    return snippet.strip()


def add_indicator(
    records,
    seen,
    candidate_id,
    sha256,
    indicator_type,
    raw_value,
    normalized_value,
    context,
    source,
):
    key = (
        candidate_id,
        indicator_type,
        normalized_value.lower(),
    )

    if key in seen:
        return

    seen.add(key)

    records.append({
        "candidate_id": candidate_id,
        "sha256": sha256,
        "indicator_type": indicator_type,
        "raw_value": raw_value,
        "normalized_value": normalized_value,
        "source": source,
        "context": context,

        # YOU fill these in manually.
        "keep": "",
        "notes": "",
    })


# ============================================================
# Load frozen candidate pool
# ============================================================

with POOL.open(
    newline="",
    encoding="utf-8",
) as f:
    pool = list(csv.DictReader(f))


records = []
summary = []


for row in pool:

    candidate_id = row["candidate_id"]

    path = (
        DATASET_ROOT
        / row["relative_path"]
    )

    if not path.exists():
        raise FileNotFoundError(path)

    sha256 = sha256_file(path)

    text = read_text(path)

    seen = set()


    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    for match in URL_RE.finditer(text):

        raw = match.group(0)

        value = clean_value(raw)

        add_indicator(
            records,
            seen,
            candidate_id,
            sha256,
            "url",
            raw,
            value,
            context_snippet(
                text,
                match.start(),
                match.end(),
            ),
            "literal_url",
        )

        # Also derive the hostname from the literal URL.
        try:
            hostname = urlparse(
                value
            ).hostname

            if hostname:
                add_indicator(
                    records,
                    seen,
                    candidate_id,
                    sha256,
                    "domain",
                    hostname,
                    hostname.lower(),
                    context_snippet(
                        text,
                        match.start(),
                        match.end(),
                    ),
                    "derived_from_url",
                )

        except Exception:
            pass


    # --------------------------------------------------------
    # IPv4
    # --------------------------------------------------------

    for match in IP_RE.finditer(text):

        raw = match.group(0)

        value = valid_ipv4(raw)

        if not value:
            continue

        add_indicator(
            records,
            seen,
            candidate_id,
            sha256,
            "ipv4",
            raw,
            value,
            context_snippet(
                text,
                match.start(),
                match.end(),
            ),
            "literal_ipv4",
        )


    # --------------------------------------------------------
    # Windows filesystem paths
    # --------------------------------------------------------

    for match in PATH_RE.finditer(text):

        raw = match.group(0)

        value = clean_value(raw)

        add_indicator(
            records,
            seen,
            candidate_id,
            sha256,
            "filesystem_path",
            raw,
            value,
            context_snippet(
                text,
                match.start(),
                match.end(),
            ),
            "literal_path",
        )


    # --------------------------------------------------------
    # Bare domains
    #
    # These are deliberately kept separate because this
    # category is much noisier than URL-derived domains.
    # --------------------------------------------------------

    for match in DOMAIN_RE.finditer(text):

        raw = match.group(0)

        value = clean_value(
            raw
        ).lower()

        add_indicator(
            records,
            seen,
            candidate_id,
            sha256,
            "domain",
            raw,
            value,
            context_snippet(
                text,
                match.start(),
                match.end(),
            ),
            "bare_domain_regex",
        )


    candidate_rows = [
        r
        for r in records
        if r["candidate_id"] == candidate_id
    ]

    summary.append({
        "candidate_id": candidate_id,
        "sha256": sha256,
        "filename": path.name,
        "review_rows": len(candidate_rows),

        "urls": sum(
            r["indicator_type"] == "url"
            for r in candidate_rows
        ),

        "domains": sum(
            r["indicator_type"] == "domain"
            for r in candidate_rows
        ),

        "ipv4": sum(
            r["indicator_type"] == "ipv4"
            for r in candidate_rows
        ),

        "filesystem_paths": sum(
            r["indicator_type"] == "filesystem_path"
            for r in candidate_rows
        ),

        # Fill this in after reviewing the sample.
        "accept_sample": "",
        "sample_notes": "",
    })


# ============================================================
# Save IOC review rows
# ============================================================

OUT_ROWS.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT_ROWS.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fields = [
        "candidate_id",
        "sha256",
        "indicator_type",
        "raw_value",
        "normalized_value",
        "source",
        "context",
        "keep",
        "notes",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(records)


# ============================================================
# Save candidate-level review summary
# ============================================================

with OUT_SUMMARY.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fields = [
        "candidate_id",
        "sha256",
        "filename",
        "review_rows",
        "urls",
        "domains",
        "ipv4",
        "filesystem_paths",
        "accept_sample",
        "sample_notes",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(summary)


print(
    "========== IOC REVIEW PACKAGE =========="
)

print(
    "Candidates:",
    len(summary),
)

print(
    "IOC review rows:",
    len(records),
)

print()
print("Review rows:")
print(OUT_ROWS)

print()
print("Candidate summary:")
print(OUT_SUMMARY)

print(
    "========================================"
)
