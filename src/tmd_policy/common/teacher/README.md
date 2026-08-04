# Teacher

`cache.py` defines `TeacherCacheIdentity`; its key includes observation content, canonical task, model/processor revisions, inference schedule, seed/sample, and the student action queried. This prevents reuse across schedules or evaluated actions. Teacher querying itself is explicit and never occurs on import.

