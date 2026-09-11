# Public datasets

## IBM AMLSim 20K samples

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
