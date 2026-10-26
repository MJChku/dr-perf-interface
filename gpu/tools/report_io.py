"""Read a standalone GPU report without loading any collector or fitter."""
import gzip
import json
from pathlib import Path


def read_report(path):
    path = Path(path)
    if path.is_dir():
        status = path/'capture-status.json'
        if status.is_file() and json.loads(status.read_text()).get('status') != 'complete':
            raise ValueError('latest GPU capture did not complete; previous report is not the latest run')
        path = path / 'profile.gpu.drperf.json'
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as source:
        report = json.load(source)
    if (not isinstance(report, dict) or report.get('schema') != 'drperf.gpu.v1'
            or report.get('backend') != 'gpu' or not isinstance(report.get('regions'), list)):
        raise ValueError(f'{path}: expected a drperf.gpu.v1 GPU report')
    return report
