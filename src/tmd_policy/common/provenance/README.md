# Provenance

`capture.py` records repository/dependency commit, full tracked and untracked binary patch plus SHA-256, status, exact command/config/seeds/revisions/task registry, Python/package/CUDA/cuDNN/GPU/host details. Dirty patches are written beside `provenance.json`; capture is read-only outside the output directory.

