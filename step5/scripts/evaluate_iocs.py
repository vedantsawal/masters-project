from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import argparse
import csv
import ipaddress
import json
import re


MP_ROOT = Path("/pscratch/sd/v/vinay/masters-project")

DEFAULT_GOLD = (
    MP_ROOT
    / "step5/data/ground_truth_iocs.jsonl"
)

DEFAULT_PRED = (
    MP_ROOT
    / "step5/results/clean_vicuna_raw.jsonl"
)

DEFAULT_PER_SAMPLE = (
    MP_ROOT
    / "step5/results/clean_eval_per_sample.csv"
)

DEFAULT_SUMMARY = (
    MP_ROOT
    / "step5/results/clean_eval_summary.json"
)


URL_RE = re.compile(
    r'https?://[^\s"\'<>\]]+',
    re.IGNORECASE,
)

MARKDOWN_LINK_RE = re.compile(
    r'\[([^\]]+)\]\((https?://[^)]+)\)',
    re.IGNORECASE,
)


def remove_markdown_escapes(value):
    # Remove backslashes used only to escape Markdown punctuation.
    return re.sub(
        r'\\([\\`*_{}\[\]()#+.!~&=-])',
        r'\1',
        value,
    )


def normalize_url(value):
    if not isinstance(value, str):
        return None

    value = value.strip()
    value = remove_markdown_escapes(value)

    # If this is a Markdown link, prefer its target.
    m = MARKDOWN_LINK_RE.search(value)

    if m:
        value = m.group(2)
    else:
        # Otherwise find an actual URL inside the string.
        urls = URL_RE.findall(value)

        if not urls:
            return None

        # Markdown often contains the same URL twice.
        value = urls[-1]

    value = remove_markdown_escapes(value)

    # Remove punctuation commonly introduced by prose/Markdown.
    value = value.rstrip('.,;\'"')

    try:
        parts = urlsplit(value)

        if parts.scheme.lower() not in {"http", "https"}:
            return None

        if not parts.netloc:
            return None

        # Lowercase scheme and hostname only.
        hostname = parts.hostname

        if hostname is None:
            return None

        netloc = hostname.lower()

        if parts.port is not None:
            netloc += f":{parts.port}"

        normalized = urlunsplit(
            (
                parts.scheme.lower(),
                netloc,
                parts.path,
                parts.query,
                parts.fragment,
            )
        )

        return normalized

    except Exception:
        return None


def normalize_ipv4(value):
    if not isinstance(value, str):
        return None

    value = value.strip()

    try:
        ip = ipaddress.ip_address(value)

        if ip.version != 4:
            return None

        return str(ip)

    except ValueError:
        return None


def strict_parse(raw):
    try:
        obj = json.loads(raw)
    except Exception:
        return None, False, False

    strict_json = True

    schema_ok = (
        isinstance(obj, dict)
        and set(obj.keys()) == {"urls", "ipv4"}
        and isinstance(obj["urls"], list)
        and isinstance(obj["ipv4"], list)
    )

    if not schema_ok:
        return obj, strict_json, False

    return obj, strict_json, True


def extract_array_body(raw, key):
    pattern = re.compile(
        rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]',
        re.IGNORECASE | re.DOTALL,
    )

    m = pattern.search(raw)

    if not m:
        return None

    return m.group(1)


def quoted_items(body):
    if body is None:
        return []

    # Sufficient for the simple IPv4 strings even when the URL
    # portion of the surrounding JSON is malformed.
    return re.findall(
        r'"([^"]*)"',
        body,
    )


def fallback_parse(raw):
    url_body = extract_array_body(
        raw,
        "urls",
    )

    ipv4_body = extract_array_body(
        raw,
        "ipv4",
    )

    urls = []

    if url_body is not None:
        urls = URL_RE.findall(
            remove_markdown_escapes(url_body)
        )

    ipv4_items = quoted_items(
        ipv4_body
    )

    return {
        "urls": urls,
        "ipv4": ipv4_items,
    }


def normalize_prediction(obj):
    urls = set()
    ipv4s = set()

    invalid_urls = set()
    invalid_ipv4 = set()

    for raw_url in obj.get("urls", []):
        normalized = normalize_url(raw_url)

        if normalized is None:
            invalid_urls.add(
                f"INVALID_URL::{raw_url}"
            )
        else:
            urls.add(normalized)

    for raw_ip in obj.get("ipv4", []):
        normalized = normalize_ipv4(raw_ip)

        if normalized is None:
            invalid_ipv4.add(
                f"INVALID_IPV4::{raw_ip}"
            )
        else:
            ipv4s.add(normalized)

    return (
        urls,
        ipv4s,
        invalid_urls,
        invalid_ipv4,
    )


def normalize_gold(record):
    urls = {
        normalize_url(x)
        for x in record["iocs"]["urls"]
    }

    urls.discard(None)

    ipv4s = {
        normalize_ipv4(x)
        for x in record["iocs"]["ipv4"]
    }

    ipv4s.discard(None)

    return urls, ipv4s


def counts(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    return tp, fp, fn


def metric(tp, fp, fn):
    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD,
    )

    parser.add_argument(
        "--pred",
        type=Path,
        default=DEFAULT_PRED,
    )

    parser.add_argument(
        "--per-sample",
        type=Path,
        default=DEFAULT_PER_SAMPLE,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load ground truth
    # ---------------------------------------------------------

    gold = {}

    with args.gold.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            record = json.loads(line)
            gold[
                record["candidate_id"]
            ] = record

    # ---------------------------------------------------------
    # Load predictions
    # ---------------------------------------------------------

    predictions = []

    with args.pred.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            predictions.append(
                json.loads(line)
            )

    if len(predictions) != 30:
        raise RuntimeError(
            f"Expected 30 predictions, "
            f"found {len(predictions)}"
        )

    # ---------------------------------------------------------
    # Score
    # ---------------------------------------------------------

    rows = []

    totals = {
        "url_tp": 0,
        "url_fp": 0,
        "url_fn": 0,
        "ipv4_tp": 0,
        "ipv4_fp": 0,
        "ipv4_fn": 0,
    }

    strict_json_count = 0
    schema_ok_count = 0
    fallback_count = 0
    exact_count = 0

    for pred_record in predictions:

        cid = pred_record["candidate_id"]

        if cid not in gold:
            raise RuntimeError(
                f"No ground truth for {cid}"
            )

        raw = pred_record["raw_output"]

        obj, strict_json, schema_ok = (
            strict_parse(raw)
        )

        if strict_json:
            strict_json_count += 1

        if schema_ok:
            schema_ok_count += 1
            parse_mode = "strict"

        else:
            obj = fallback_parse(raw)
            fallback_count += 1
            parse_mode = "fallback"

        (
            pred_urls,
            pred_ipv4,
            invalid_urls,
            invalid_ipv4,
        ) = normalize_prediction(obj)

        gold_urls, gold_ipv4 = (
            normalize_gold(gold[cid])
        )

        url_tp, url_fp, url_fn = counts(
            pred_urls,
            gold_urls,
        )

        ip_tp, ip_fp, ip_fn = counts(
            pred_ipv4,
            gold_ipv4,
        )

        # Invalid typed predictions are model errors and
        # therefore count as false positives.
        url_fp += len(invalid_urls)
        ip_fp += len(invalid_ipv4)

        totals["url_tp"] += url_tp
        totals["url_fp"] += url_fp
        totals["url_fn"] += url_fn

        totals["ipv4_tp"] += ip_tp
        totals["ipv4_fp"] += ip_fp
        totals["ipv4_fn"] += ip_fn

        exact = (
            pred_urls == gold_urls
            and pred_ipv4 == gold_ipv4
            and not invalid_urls
            and not invalid_ipv4
        )

        if exact:
            exact_count += 1

        rows.append({
            "pilot_id": pred_record["pilot_id"],
            "candidate_id": cid,
            "parse_mode": parse_mode,
            "strict_json": strict_json,
            "schema_ok": schema_ok,

            "gold_urls": len(gold_urls),
            "pred_urls": len(pred_urls),
            "url_tp": url_tp,
            "url_fp": url_fp,
            "url_fn": url_fn,

            "gold_ipv4": len(gold_ipv4),
            "pred_ipv4": len(pred_ipv4),
            "ipv4_tp": ip_tp,
            "ipv4_fp": ip_fp,
            "ipv4_fn": ip_fn,

            "invalid_url_predictions":
                len(invalid_urls),

            "invalid_ipv4_predictions":
                len(invalid_ipv4),

            "exact_match": exact,

            "gold_url_values":
                json.dumps(
                    sorted(gold_urls)
                ),

            "pred_url_values":
                json.dumps(
                    sorted(pred_urls)
                ),

            "gold_ipv4_values":
                json.dumps(
                    sorted(gold_ipv4)
                ),

            "pred_ipv4_values":
                json.dumps(
                    sorted(pred_ipv4)
                ),

            "invalid_values":
                json.dumps(
                    sorted(
                        invalid_urls
                        | invalid_ipv4
                    )
                ),
        })

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    url_p, url_r, url_f1 = metric(
        totals["url_tp"],
        totals["url_fp"],
        totals["url_fn"],
    )

    ip_p, ip_r, ip_f1 = metric(
        totals["ipv4_tp"],
        totals["ipv4_fp"],
        totals["ipv4_fn"],
    )

    combined_tp = (
        totals["url_tp"]
        + totals["ipv4_tp"]
    )

    combined_fp = (
        totals["url_fp"]
        + totals["ipv4_fp"]
    )

    combined_fn = (
        totals["url_fn"]
        + totals["ipv4_fn"]
    )

    micro_p, micro_r, micro_f1 = metric(
        combined_tp,
        combined_fp,
        combined_fn,
    )

    summary = {
        "samples": len(predictions),

        "strict_json": strict_json_count,
        "correct_schema": schema_ok_count,
        "fallback_parsed": fallback_count,

        "format_compliance_rate":
            schema_ok_count / len(predictions),

        "url": {
            "tp": totals["url_tp"],
            "fp": totals["url_fp"],
            "fn": totals["url_fn"],
            "precision": url_p,
            "recall": url_r,
            "f1": url_f1,
        },

        "ipv4": {
            "tp": totals["ipv4_tp"],
            "fp": totals["ipv4_fp"],
            "fn": totals["ipv4_fn"],
            "precision": ip_p,
            "recall": ip_r,
            "f1": ip_f1,
        },

        "combined_micro": {
            "tp": combined_tp,
            "fp": combined_fp,
            "fn": combined_fn,
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
        },

        "exact_match_samples": exact_count,

        "exact_match_accuracy":
            exact_count / len(predictions),

        "evaluation_policy": {
            "primary_iocs":
                "manually validated URLs and IPv4 addresses",

            "strict_format":
                "JSON object with exactly urls and ipv4 list keys",

            "fallback":
                "deterministic extraction from malformed response",

            "invalid_typed_predictions":
                "counted as false positives",

            "duplicates":
                "deduplicated within IOC type",

            "manual_repairs":
                False,
        },
    }

    # ---------------------------------------------------------
    # Write artifacts
    # ---------------------------------------------------------

    args.per_sample.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.per_sample.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    with args.summary.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

        f.write("\n")

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print(
        "========== CLEAN VICUNA IOC BASELINE =========="
    )

    print(
        f"Samples: {len(predictions)}"
    )

    print(
        f"Strict JSON / correct schema: "
        f"{schema_ok_count}/{len(predictions)} "
        f"({100 * summary['format_compliance_rate']:.2f}%)"
    )

    print(
        f"Fallback-parsed responses: "
        f"{fallback_count}"
    )

    print()

    print("URL metrics:")
    print(
        f"  TP={totals['url_tp']} "
        f"FP={totals['url_fp']} "
        f"FN={totals['url_fn']}"
    )
    print(
        f"  Precision={url_p:.4f} "
        f"Recall={url_r:.4f} "
        f"F1={url_f1:.4f}"
    )

    print()

    print("IPv4 metrics:")
    print(
        f"  TP={totals['ipv4_tp']} "
        f"FP={totals['ipv4_fp']} "
        f"FN={totals['ipv4_fn']}"
    )
    print(
        f"  Precision={ip_p:.4f} "
        f"Recall={ip_r:.4f} "
        f"F1={ip_f1:.4f}"
    )

    print()

    print("Combined micro metrics:")
    print(
        f"  TP={combined_tp} "
        f"FP={combined_fp} "
        f"FN={combined_fn}"
    )
    print(
        f"  Precision={micro_p:.4f} "
        f"Recall={micro_r:.4f} "
        f"F1={micro_f1:.4f}"
    )

    print()

    print(
        f"Exact-match samples: "
        f"{exact_count}/{len(predictions)} "
        f"({100 * summary['exact_match_accuracy']:.2f}%)"
    )

    print()
    print(
        "Per-sample results:",
        args.per_sample,
    )
    print(
        "Summary:",
        args.summary,
    )

    print(
        "==============================================="
    )


if __name__ == "__main__":
    main()
