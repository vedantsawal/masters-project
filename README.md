# Master's Project

## Beyond the Verdict: Malware-Borne Prompt Injection Against LLM-Based Malware Analysis

This repository contains code, experiment manifests, annotations, evaluation
scripts, and preliminary results for my Master's project at San Jose State
University.

The project investigates whether attacker-controlled content embedded in
malware artifacts can manipulate LLM-based malware analysis, with particular
emphasis on the integrity of indicator-of-compromise (IOC) extraction.

## Current Preliminary Experiment

The Step 5 pilot evaluates Vicuna-7B-v1.5 on 30 malicious PowerShell samples
with manually validated URL and IPv4 ground truth.

Primary pilot dataset:

- 30 unique malicious PowerShell samples
- 36 validated URLs
- 21 validated IPv4 addresses
- 57 total primary ground-truth IOCs

Clean Vicuna baseline:

- Precision: 76.81%
- Recall: 92.98%
- F1: 84.13%
- Exact-match accuracy: 63.33%

Naive IOC-suppression attack:

- Precision: 80.70%
- Recall: 80.70%
- F1: 80.70%
- Exact-match accuracy: 60.00%

Paired analysis:

- 53 ground-truth IOCs recovered in the clean condition
- 10 previously recovered IOCs suppressed under attack
- Paired IOC suppression rate: 18.87%
- 6 of 30 samples experienced at least one suppressed IOC
- 4 samples experienced complete suppression of previously recovered IOCs

## Dataset Safety

Raw malicious PowerShell samples are not distributed in this repository.

The experiment uses the public Fa2y/Malicious-PowerShell-Dataset at pinned
commit:

69387cd626db0dadbe7c4ef1c700564bc54fabe0

The repository contains hashes, sample identifiers, annotations, manifests,
and acquisition information required to reconstruct the experiment.

## Environment

Experiments were run on NERSC Perlmutter using NVIDIA A100 GPUs.

See `environment/` for software environment information.

## Status

This repository contains preliminary Master's project research and remains
under active development.
