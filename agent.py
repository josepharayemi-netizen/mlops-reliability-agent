"""Evidence-based reliability monitoring for binary classification models."""
import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULTS = {'minimum_samples': 30, 'drift_tv': 0.25, 'accuracy_drop': 0.05,
            'recall_drop': 0.10, 'missing_rate_increase': 0.10,
            'p95_latency_ms': 500.0, 'error_rate': 0.05, 'minimum_label_coverage': 0.8}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(document):
    if not isinstance(document, dict):
        raise ValueError('Expected a JSON object.')
    if not isinstance(document.get('model'), str) or not document['model'].strip():
        raise ValueError('model must be a nonempty name/version string.')
    features = document.get('features')
    if not isinstance(features, list) or not features or len(features) > 100:
        raise ValueError('features must contain 1–100 numeric feature names.')
    if any(not isinstance(f, str) or not f for f in features) or len(set(features)) != len(features):
        raise ValueError('Feature names must be nonempty unique strings.')
    supplied = document.get('thresholds', {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError('Unknown thresholds; see README for supported names.')
    config = DEFAULTS | supplied
    for key, value in config.items():
        if not number(value):
            raise ValueError(f'{key} must be finite and numeric.')
        if key == 'minimum_samples':
            if not isinstance(value, int) or value < 2:
                raise ValueError('minimum_samples must be an integer >= 2.')
        elif key == 'p95_latency_ms':
            if value <= 0:
                raise ValueError('p95_latency_ms must be positive.')
        elif not 0 < value <= 1:
            raise ValueError(f'{key} must be in (0, 1].')
    for name in ('baseline', 'current'):
        rows = document.get(name)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 20000:
            raise ValueError(f'{name} must contain 1–20,000 rows.')
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get('features'), dict):
                raise ValueError(f'{name}[{index}] must contain a features object.')
            for f in features:
                value = row['features'].get(f)
                if value is not None and not number(value):
                    raise ValueError(f'{name}[{index}].features.{f} must be finite numeric or null.')
            for field in ('prediction', 'actual'):
                value = row.get(field)
                if value is not None and (type(value) is not int or value not in (0, 1)):
                    raise ValueError(f'{field} must be integer 0, integer 1, or null.')
            if row.get('latency_ms') is not None and (not number(row['latency_ms']) or row['latency_ms'] < 0):
                raise ValueError('latency_ms must be nonnegative and finite or null.')
            if row.get('error') is not None and type(row['error']) is not bool:
                raise ValueError('error must be boolean or null.')
    return config


def distribution_drift(reference, current):
    """Total variation of common reference-derived bins, with explicit tail bins."""
    if not reference or not current:
        return None
    low, high = min(reference), max(reference)
    if low == high:
        def bucket(x):
            return -1 if x < low else 1 if x > high else 0
    else:
        def bucket(x):
            if x < low:
                return -1
            if x > high:
                return 10
            # Scale operands before subtracting to avoid overflow at extreme values.
            scale = max(abs(low), abs(high), 1)
            fraction = (x / scale - low / scale) / (high / scale - low / scale)
            return min(9, int(fraction * 10))
    a, b = Counter(map(bucket, reference)), Counter(map(bucket, current))
    return min(1.0, sum(abs(a[k] / len(reference) - b[k] / len(current)) for k in a.keys() | b.keys()) / 2)


def metrics(rows):
    labelled = [r for r in rows if r.get('actual') is not None and r.get('prediction') is not None]
    tp = sum(r['actual'] == r['prediction'] == 1 for r in labelled)
    tn = sum(r['actual'] == r['prediction'] == 0 for r in labelled)
    fp = sum(r['actual'] == 0 and r['prediction'] == 1 for r in labelled)
    fn = sum(r['actual'] == 1 and r['prediction'] == 0 for r in labelled)
    latencies = sorted(r['latency_ms'] for r in rows if r.get('latency_ms') is not None)
    errors = [r['error'] for r in rows if r.get('error') is not None]
    return {'rows': len(rows), 'labelled_rows': len(labelled), 'label_coverage': len(labelled) / len(rows),
            'accuracy': (tp + tn) / len(labelled) if labelled else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'positive_labels': tp + fn, 'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'p95_latency_ms': latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None,
            'latency_samples': len(latencies), 'error_rate': sum(errors) / len(errors) if errors else None,
            'error_samples': len(errors)}


def assess(document):
    config = validate(document)
    baseline, current = document['baseline'], document['current']
    before, after = metrics(baseline), metrics(current)
    findings, checks, drift = [], [], []

    def check(id, title, eligible, observed, threshold, severity, recommendation):
        status = 'unknown' if not eligible else 'alert' if observed >= threshold else 'ok'
        checks.append({'id': id, 'title': title, 'status': status, 'value': observed, 'threshold': threshold})
        if status == 'alert':
            findings.append({'id': id, 'severity': severity, 'title': title, 'evidence': f'{observed:.6g} >= configured threshold {threshold:.6g}', 'recommendation': recommendation})
        elif status == 'unknown':
            findings.append({'id': id, 'severity': 'review', 'title': title + ': insufficient evidence', 'evidence': 'Minimum sample, label coverage, or metric availability requirement not met.', 'recommendation': 'Collect representative observations and labels before deciding whether the model is healthy.'})

    minimum = config['minimum_samples']
    for feature in document['features']:
        a = [r['features'][feature] for r in baseline if r['features'].get(feature) is not None]
        b = [r['features'][feature] for r in current if r['features'].get(feature) is not None]
        missing_before, missing_after = 1-len(a)/len(baseline), 1-len(b)/len(current)
        tv = distribution_drift(a, b)
        drift.append({'feature': feature, 'tv': tv, 'baseline_missing_rate': missing_before, 'current_missing_rate': missing_after,
                      'baseline_samples': len(a), 'current_samples': len(b)})
        check('drift:' + feature, 'Distribution change: ' + feature, min(len(a), len(b)) >= minimum, tv, config['drift_tv'], 'warning',
              'Compare collection logic, traffic segments and feature transformations with the reference window. Drift alone does not prove model degradation; validate on labelled data before retraining.')
        check('missing:' + feature, 'Missing-value increase: ' + feature, min(len(baseline),len(current)) >= minimum,
              missing_after-missing_before, config['missing_rate_increase'], 'warning',
              'Inspect the upstream data contract, ingestion changes and missing-value handling. Confirm source availability before changing the model.')
    performance_ready = min(before['labelled_rows'],after['labelled_rows']) >= minimum and min(before['label_coverage'],after['label_coverage']) >= config['minimum_label_coverage']
    for metric in ('accuracy','recall'):
        available = before[metric] is not None and after[metric] is not None
        enough = performance_ready and (metric != 'recall' or min(before['positive_labels'],after['positive_labels']) >= minimum)
        check('performance:' + metric, metric.title() + ' decline', available and enough,
              before[metric]-after[metric] if available else None, config[metric+'_drop'], 'critical',
              'Check label maturity, class balance and affected segments. Compare a candidate or previous model on a held-out labelled dataset; request human approval before deployment or rollback.')
    check('operations:latency', 'P95 inference latency', after['latency_samples'] >= minimum, after['p95_latency_ms'], config['p95_latency_ms'], 'warning',
          'Inspect request sizes, serving logs, resource saturation and recent deployments. Compare changes in a staging environment.')
    check('operations:errors', 'Inference error rate', after['error_samples'] >= minimum, after['error_rate'], config['error_rate'], 'critical',
          'Inspect failed-request logs and dependency health. Check recent releases and prepare a reviewed recovery plan.')
    findings.sort(key=lambda f: ({'critical':0,'warning':1,'review':2}[f['severity']],f['id']))
    status = 'action_required' if any(f['severity']=='critical' for f in findings) else 'investigate' if any(f['severity']=='warning' for f in findings) else 'insufficient_evidence' if findings else 'within_thresholds'
    return {'model': document['model'], 'generated_at': datetime.now(timezone.utc).isoformat(), 'status':status,
            'mode':'Deterministic monitoring with optional local AI explanation', 'thresholds':config,
            'evidence_sha256':hashlib.sha256(json.dumps(document,sort_keys=True,allow_nan=False).encode()).hexdigest(),
            'metrics':{'baseline':before,'current':after},'drift':drift,'checks':checks,'findings':findings,
            'limitations':'Snapshot analysis for numeric features and binary classification only. Threshold alerts are heuristics, not statistical significance tests. Data drift does not establish concept drift. No retraining, rollback or cloud changes are performed.',
            'trace':[{'step':'validate','result':'Input schema and numeric values verified'}, {'step':'measure','result':'Compared feature distributions, labelled performance and operational measurements'}, {'step':'plan','result':f'{len(findings)} prioritized investigation items; execution remains with a human reviewer'}]}


def markdown(report):
    lines = ['# MLOps Reliability Assessment', '', 'Model: '+report['model'], 'Status: '+report['status'],
             'Generated: '+report['generated_at'], 'Evidence SHA-256: '+report['evidence_sha256'], '', report['limitations'], '']
    for f in report['findings']:
        lines += [f"## [{f['severity']}] {f['title']}", f"Check: {f['id']}", f['evidence'], '', f['recommendation'], '']
    if not report['findings']:
        lines += ['All assessed checks are within configured thresholds. This is not a guarantee of model quality.']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,default=Path('report.json'))
    parser.add_argument('--markdown',type=Path)
    args=parser.parse_args()
    try:
        report=assess(json.loads(args.input.read_text(encoding='utf-8')))
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        if args.markdown: args.markdown.write_text(markdown(report),encoding='utf-8')
        print(f"{report['status']}: {len(report['findings'])} investigation items. Saved {args.output}")
    except (ValueError,OSError) as error:
        parser.exit(1,f'Assessment failed: {error}\n')
