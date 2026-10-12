# Public datasets

## IBM AML-Data: current transaction detection task

Primary source: <https://github.com/IBM/AML-Data>. The public AMLworld
transactions are synthetic and contain transaction-level laundering labels.
Review the upstream data license before downloading or redistribution.
Suggested local paths are `data/raw/ibm/HI-Small_Trans.csv` and
`data/raw/ibm/LI-Small_Trans.csv`. No IBM AML-Data files were downloaded as part
of the v0.3 implementation. HI/LI configurations share one generator.

On 2026-10-09 (Asia/Shanghai), the official Kaggle **version 8**
HI-Small transactions and patterns were downloaded for v0.5 validation.
Source metadata, license declarations and hashes are preserved in
`data/ibm_aml_manifest.json`; original data remain ignored in `data/raw/ibm/`.
Local `provenance.json` contains download verification, `prefix_profile.json`
contains fixed-prefix diagnostics, and `full_time_profile.json` contains a
streaming full-file date/hour quality profile. LI-Small was not downloaded.

The full HI-Small file has 5,078,345 records and 5,177 positive labels.
Its source rows are not in chronological order. The v0.5 20k/50k/100k source
prefixes cover only the first 30 minutes and all have no positive test
records under the fixed timestamp-group split. The 50k benchmark is therefore
an engineering pilot; AP and recall are undefined, not zero. The historical
v0.5 `full_time_profile.json` checks CSV structure, timestamps and labels. It
did not validate every amount or participant, audit business duplicates, or
map transactions to injected laundering chains.

The v0.6 preparation has now completed strict adapter validation over all
5,078,345 source rows, retaining all 5,177 positive labels, 5,073,168 negative
labels and no unlabeled rows. It validates required columns and row widths,
timestamps, nonempty sender/receiver banks and accounts, payment/receiving
currencies and payment format, finite positive paid/received amounts, and
binary labels when present. Its completed manifest is
`outputs/ibm-hi-small-v8-prepared-v06/manifest.json`; the disk-backed
`transactions.sqlite` stores the original source row numbers and transaction
IDs, with labels in an independent table. The manifest records the complete
source-file SHA-256 and database SHA-256. Transactions are read in UTC
timestamp/event-ID order without constructing a full in-memory case graph.
This validation does not establish that business records are unique or that
injected-chain membership is correct; those full-data audits remain pending.

A declared view of the earliest six UTC hours,
`[2022-09-01T00:00:00Z, 2022-09-01T06:00:00Z)`, has completed causal cache
construction at `outputs/ibm-hi-small-v8-first6h-cache-v06/manifest.json`.
It contains 511,357 transactions and 75 positive labels. Complete timestamp
groups are split chronologically: training has 305,397 transactions/15
positives, validation 103,522/33, and test 102,438/27. Selection uses time
bounds, without filtering by label. A time view starts with empty history;
rows before its lower boundary are not replayed. History continues across
the train/validation/test boundaries within that view, using strictly earlier
transactions and keeping labels out of feature construction. A completed
six-hour cache is data preparation, not completed full-file model training or
publication-quality evaluation of all 5,078,345 transactions.

The six-hour standard pilot (15 models) and fixed-periodic numerical
diagnostic (9 neural models) are now complete. These are exploratory results,
not full-file training or evidence of an event-model advantage. See
[the experiment record](../docs/experiment_progress.md) for results and
restoration checks.

The detector reads the original duplicate `Account` header correctly. Labels
are stored only in evaluation metadata for in-memory cases, or independent
`labels`/`evaluation_labels` tables for prepared datasets and causal caches;
missing labels remain missing. See the root README for causal
train/validation/test and scoring commands.

The built-in `detect-demo` CSV is a small locally generated smoke fixture. It
is not IBM AMLworld and its scores cannot be used as publication evidence.

## IBM AMLSim 20K samples

Historical node-level diagnostic data; `isFraud` is not a transaction-level
laundering label. The new supervised detector does not manufacture transaction
labels from these node labels.

Source: <https://github.com/IBM/AMLSim/tree/master/sample>

Downloaded samples:

- `20K_cycle200.tgz`: 20,000 accounts, 117,805 transactions, 200 injected cycles.
- `20K_fanin200.tgz`: 20,000 accounts, 118,250 transactions, 200 injected fan-in patterns.
- `20K_fanin200cycle200.tgz`: 20,000 accounts, 120,558 transactions, 200 cycles and 200 fan-in patterns.

The upstream repository is distributed under Apache License 2.0. The files in
`data/raw/` are ignored by version control so the research repository records
the source and checksums without silently redistributing third-party data.

SHA-256 checksums for the downloaded archives are recorded in
`data/amlsim_checksums.sha256`.

The node columns `isFraud` and `fraudStep` are evaluation labels. They must not
be copied into entity attributes, atomic-event attributes, graph topology, or
candidate-generation features.
