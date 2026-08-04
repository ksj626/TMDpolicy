# Methods

`base.py` defines the typed research interface and `DryRunReport`; `registry.py` dispatches five strictly named methods plus the occupancy diagnostic. Each method owns its models, losses, optimizer schedule, checkpoint, capabilities, and sampling. Shared comparison infrastructure is under `common/`; mathematical contracts are under `docs/methods/`.

