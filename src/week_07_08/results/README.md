# Retained vision evidence

These files document real Pi experiments; no detector or camera inference was
rerun while preparing the documentation.

| Record | Origin and interpretation |
|---|---|
| `recorded_observations.json` | All 38 observation/camera metadata records from the supplied `observations.zip`; includes derived timing statistics and the two first-call records |
| `reported_yolo26_comparison.json` | Terminal measurements reported for the table image; original benchmark JSON files were not supplied |
| `glasses-held/` | Original run `20261015T023150_e3cccfc5`: glasses visible but no retained glasses detection |
| `glasses-worn/` | Original run `20261015T024744_1856c588`: glasses score 0.8966 |
| `glasses-lowered/` | Original run `20261015T024844_4f8d5d3a`: glasses score 0.2666 |
| `glasses-near-face/` | Original run `20261015T024959_03da2666`: glasses score 0.7427 near the mouth |

Each selected directory keeps the original `scene.jpg`, `scene.json`,
`annotated.jpg`, `observation.json` and `metrics.json`. The original receipt times,
run IDs and local source paths are retained. These historical timestamps must not
pass live observation freshness checks.

The aggregated timing statistics exclude frames with
`prediction_index_in_process == 1` (one per process). For the remaining 36 calls,
medians use `statistics.median` and p95 uses linear interpolation at
`(n - 1) * 0.95` in sorted samples. Model construction is recorded per instance,
not incurred again on every call. Neither these records nor the ten-repeat YOLO26
measurements are a general accuracy benchmark.

To independently reproduce the three medians from the committed records:

```bash
python3 - <<'PY'
import json
from pathlib import Path
import statistics

data = json.loads(Path("src/week_07_08/results/recorded_observations.json").read_text())
steady = [f["observation"]["metrics"] for f in data["frames"]
          if f["observation"]["metrics"]["prediction_index_in_process"] != 1]
for field in ("prediction_ms", "capture_process_ms", "observe_scene_total_ms"):
    print(field, statistics.median(row[field] for row in steady))
PY
```

The main [week guide](../README.md) pairs images with commands, scores,
interpretations and the limits of the model comparisons.
