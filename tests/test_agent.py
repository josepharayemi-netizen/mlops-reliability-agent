import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from agent import assess,distribution_drift,metrics
from advisor import explain
ROOT=Path(__file__).parents[1]
def sample(name='healthy'):
    return json.loads((ROOT/'examples'/f'{name}.json').read_text())

class ReliabilityTests(unittest.TestCase):
    def test_identical_windows(self):
        r=assess(sample())
        self.assertEqual(r['status'],'within_thresholds')
        self.assertEqual(r['findings'],[])
        self.assertTrue(all(d['tv']==0 for d in r['drift']))
    def test_degradation_plan(self):
        r=assess(sample('degraded'))
        self.assertEqual(r['status'],'action_required')
        ids={f['id'] for f in r['findings']}
        self.assertEqual(ids,{'drift:transaction_amount','missing:account_age_days','performance:accuracy','performance:recall','operations:latency','operations:errors'})
        self.assertEqual(r['metrics']['current']['accuracy'],.75)
        self.assertEqual(r['metrics']['current']['recall'],.5)
    def test_unlabelled_is_unknown(self):
        r=assess(sample('unlabelled'))
        self.assertEqual(r['status'],'insufficient_evidence')
        self.assertIsNone(r['metrics']['current']['accuracy'])
        self.assertEqual(r['metrics']['current']['label_coverage'],0)
    def test_small_samples(self):
        d=sample();d['current']=d['current'][:2]
        self.assertEqual(assess(d)['status'],'insufficient_evidence')
    def test_low_label_coverage(self):
        d=sample()
        for r in d['current'][:40]:r['actual']=None
        c={x['id']:x for x in assess(d)['checks']}
        self.assertEqual(c['performance:accuracy']['status'],'unknown')
    def test_missingness_separate_from_drift(self):
        d=sample()
        for row in d['current']:row['features']['transaction_amount']=None
        c={x['id']:x for x in assess(d)['checks']}
        self.assertEqual(c['drift:transaction_amount']['status'],'unknown')
        self.assertEqual(c['missing:transaction_amount']['status'],'alert')
    def test_histogram_tails_and_constants(self):
        self.assertEqual(distribution_drift([1]*50,[1]*50),0)
        self.assertEqual(distribution_drift([1]*50,[2]*50),1)
        self.assertAlmostEqual(distribution_drift([1,2,3],[4,5,6]),1)
        self.assertEqual(distribution_drift([0,1,2],[0,1,2]),0)
        self.assertIsNone(distribution_drift([],[]))
    def test_confusion_matrix(self):
        m=metrics([{'actual':a,'prediction':p} for a,p in [(1,1),(1,0),(0,1),(0,0)]])
        self.assertEqual(m['confusion_matrix'],{'tp':1,'tn':1,'fp':1,'fn':1})
        self.assertEqual(m['precision'],.5);self.assertEqual(m['recall'],.5);self.assertEqual(m['f1'],.5)
    def test_no_positive_class(self):
        d=sample()
        for row in d['baseline']+d['current']:row['actual']=row['prediction']=0
        r=assess(d)
        self.assertIsNone(r['metrics']['current']['recall'])
        self.assertEqual(r['status'],'insufficient_evidence')
    def test_no_operational_data(self):
        d=sample()
        for row in d['current']:row.pop('latency_ms');row.pop('error')
        r=assess(d)
        self.assertEqual(r['status'],'insufficient_evidence')
    def test_percentile_nearest_rank(self):
        m=metrics([{'latency_ms':i} for i in range(1,101)])
        self.assertEqual(m['p95_latency_ms'],95)
    def test_reject_bad_values(self):
        for value in [float('nan'),float('inf'),'20',True]:
            d=sample();d['current'][0]['features']['transaction_amount']=value
            with self.assertRaises(ValueError):assess(d)
        for key,value in [('prediction',True),('actual',2),('error',1),('latency_ms',-1)]:
            d=sample();d['current'][0][key]=value
            with self.assertRaises(ValueError):assess(d)
    def test_invalid_thresholds(self):
        for value in [{'minimum_samples':1},{'drift_tv':0},{'accuracy_drop':2},{'mystery':1}]:
            d=sample();d['thresholds']=value
            with self.assertRaises(ValueError):assess(d)
    def test_exact_threshold_alert(self):
        d=sample();d['thresholds']={'p95_latency_ms':134}
        c={x['id']:x for x in assess(d)['checks']}
        self.assertEqual(c['operations:latency']['status'],'alert')
    def test_evidence_digest(self):
        d=sample();digest=assess(d)['evidence_sha256']
        self.assertEqual(digest,assess(json.loads(json.dumps(d,sort_keys=True)))['evidence_sha256'])
        d['model']='changed';self.assertNotEqual(digest,assess(d)['evidence_sha256'])
    @patch('advisor.urlopen')
    def test_local_ai_adapter(self,request):
        response=MagicMock();response.__enter__.return_value.read.return_value=b'{"message":{"content":"Review performance:recall."}}';request.return_value=response
        r=explain(assess(sample('degraded')),'What changed?','test-model')
        self.assertIn('performance:recall',r['answer'])
        payload=json.loads(request.call_args.args[0].data)
        self.assertNotIn('tools',payload)
        self.assertIn('untrusted data',payload['messages'][0]['content'])
        self.assertEqual(request.call_args.args[0].full_url,'http://127.0.0.1:11434/api/chat')

if __name__=='__main__':unittest.main()
