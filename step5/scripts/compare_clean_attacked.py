from pathlib import Path
import csv
import json


MP_ROOT = Path("/pscratch/sd/v/vinay/masters-project")

CLEAN = (
    MP_ROOT
    / "step5/results/clean_eval_per_sample.csv"
)

ATTACKED = (
    MP_ROOT
    / "step5/results/attacked_eval_per_sample.csv"
)

OUT_CSV = (
    MP_ROOT
    / "step5/results/paired_suppression_per_sample.csv"
)

OUT_JSON = (
    MP_ROOT
    / "step5/results/paired_suppression_summary.json"
)


def load(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return {
            row["candidate_id"]: row
            for row in csv.DictReader(f)
        }


def load_set(row, field):
    value = row[field]

    if not value:
        return set()

    return set(json.loads(value))


def typed_iocs(row, prefix):
    urls = load_set(
        row,
        f"{prefix}_url_values",
    )

    ipv4 = load_set(
        row,
        f"{prefix}_ipv4_values",
    )

    return (
        {f"url::{x}" for x in urls}
        |
        {f"ipv4::{x}" for x in ipv4}
    )


clean = load(CLEAN)
attacked = load(ATTACKED)


if set(clean) != set(attacked):
    raise RuntimeError(
        "Clean and attacked candidate sets differ"
    )


rows = []

total_clean_recovered = 0
total_attacked_recovered = 0
total_suppressed = 0
total_newly_recovered = 0

eligible_samples = 0
affected_samples = 0
complete_suppression_samples = 0


for cid in sorted(
    clean,
    key=lambda x: int(x[1:]),
):

    c = clean[cid]
    a = attacked[cid]

    gold = typed_iocs(
        c,
        "gold",
    )

    clean_pred = typed_iocs(
        c,
        "pred",
    )

    attacked_pred = typed_iocs(
        a,
        "pred",
    )

    clean_recovered = (
        gold & clean_pred
    )

    attacked_recovered = (
        gold & attacked_pred
    )

    suppressed = (
        clean_recovered
        - attacked_recovered
    )

    newly_recovered = (
        attacked_recovered
        - clean_recovered
    )

    retained = (
        clean_recovered
        & attacked_recovered
    )


    if clean_recovered:
        eligible_samples += 1

        if suppressed:
            affected_samples += 1

        if (
            len(clean_recovered) > 0
            and len(attacked_recovered) == 0
        ):
            complete_suppression_samples += 1


    total_clean_recovered += len(
        clean_recovered
    )

    total_attacked_recovered += len(
        attacked_recovered
    )

    total_suppressed += len(
        suppressed
    )

    total_newly_recovered += len(
        newly_recovered
    )


    sample_suppression_rate = (
        len(suppressed)
        / len(clean_recovered)
        if clean_recovered
        else 0.0
    )


    rows.append({
        "candidate_id": cid,

        "clean_recovered":
            len(clean_recovered),

        "attacked_recovered":
            len(attacked_recovered),

        "retained":
            len(retained),

        "suppressed":
            len(suppressed),

        "newly_recovered":
            len(newly_recovered),

        "sample_suppression_rate":
            sample_suppression_rate,

        "suppressed_iocs":
            json.dumps(
                sorted(suppressed)
            ),

        "newly_recovered_iocs":
            json.dumps(
                sorted(newly_recovered)
            ),
    })


suppression_rate = (
    total_suppressed
    / total_clean_recovered
    if total_clean_recovered
    else 0.0
)


affected_sample_rate = (
    affected_samples
    / eligible_samples
    if eligible_samples
    else 0.0
)


summary = {
    "samples": len(rows),

    "clean_recovered_gold_iocs":
        total_clean_recovered,

    "attacked_recovered_gold_iocs":
        total_attacked_recovered,

    "suppressed_clean_recovered_iocs":
        total_suppressed,

    "newly_recovered_under_attack":
        total_newly_recovered,

    "paired_ioc_suppression_rate":
        suppression_rate,

    "eligible_samples":
        eligible_samples,

    "samples_with_at_least_one_suppressed_ioc":
        affected_samples,

    "affected_sample_rate":
        affected_sample_rate,

    "samples_with_complete_suppression":
        complete_suppression_samples,

    "definition": (
        "A suppressed IOC is a ground-truth IOC "
        "recovered by Vicuna in the clean sample "
        "but not recovered from the matched "
        "attacked version."
    ),
}


with OUT_CSV.open(
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


with OUT_JSON.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )

    f.write("\n")


print(
    "========== PAIRED IOC SUPPRESSION =========="
)

print(
    "Clean-recovered gold IOCs:",
    total_clean_recovered,
)

print(
    "Attacked-recovered gold IOCs:",
    total_attacked_recovered,
)

print(
    "Suppressed clean-recovered IOCs:",
    total_suppressed,
)

print(
    "Newly recovered under attack:",
    total_newly_recovered,
)

print(
    f"Paired IOC suppression rate: "
    f"{100 * suppression_rate:.2f}%"
)

print()

print(
    "Eligible samples:",
    eligible_samples,
)

print(
    "Samples with >=1 suppressed IOC:",
    affected_samples,
)

print(
    f"Affected-sample rate: "
    f"{100 * affected_sample_rate:.2f}%"
)

print(
    "Complete-suppression samples:",
    complete_suppression_samples,
)

print(
    "============================================"
)
