# Config

`research.py` loads YAML into `ResearchConfig`, validates method names, required sections, positive steps/resources, and immutable 40-character revisions. It does not merge hidden defaults: every value printed by dry-run is the consumed value.

