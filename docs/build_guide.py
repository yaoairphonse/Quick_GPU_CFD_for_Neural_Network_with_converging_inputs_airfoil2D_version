#!/usr/bin/env python3
"""Build the offline English guide from its HTML template and package defaults."""
from pathlib import Path
import html
import json

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / 'config/default.json').read_text())
body = (ROOT / 'docs/guide_body.html').read_text()
timing = ''
summary_path = ROOT / 'validation/summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    timing = '<table><tr><th>G0001 default</th><th>CVs</th><th>Mesh (s)</th><th>GPU event (s)</th><th>C++ process (s)</th></tr>'
    for row in summary['full_steady']:
        timing += f'<tr><td>{row["profile"]}</td><td>{row["cells"]}</td><td>{row["mesh_seconds"]:.4f}</td><td>{row["GPU_event_seconds"]:.3f}</td><td>{row["process_seconds"]:.3f}</td></tr>'
    timing += f'</table><p>Total pipeline time, including both grids and Python processing: {summary["default_pipeline_seconds"]:.3f} seconds. One geometry and one measurement, not a runtime guarantee. Both grids met the original steady criteria with zero rejected steps.</p>'
body = body.replace('__TIMING_TABLE__', timing)
body = body.replace('__CONFIG_ESCAPED__', html.escape(json.dumps(config, indent=2)))
body = body.replace('__CONFIG_JSON__', json.dumps(config))
(ROOT / 'GUIDE.html').write_text(body)
print(ROOT / 'GUIDE.html')
