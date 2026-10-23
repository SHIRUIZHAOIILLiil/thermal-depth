"""Read-only paired smoke or final validation report; no automatic submissions."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.train_da2_residual import pair_smoke
from tools.da2_continuation import read


def report(base):
    dirs=[base/f'{arm}_s43' for arm in ('frame','fixed')]
    contracts=[read(d/'contract.json') for d in dirs]
    if [c['arm'] for c in contracts]!=['frame','fixed']: raise ValueError('Wrong arms')
    if {k:v for k,v in contracts[0].items() if k!='arm'}!={k:v for k,v in contracts[1].items() if k!='arm'}:
        raise ValueError('Paired recipes differ')
    logs=[]; results=[]
    for d in dirs:
        done=read(d/'completed.json')
        if done['mode']!='train' or done['step']!=20000: raise ValueError('Incomplete run')
        logs.append({r['step']:r for r in map(json.loads,(d/'train.jsonl').read_text().splitlines())})
        if any(s not in logs[-1] or logs[-1][s]['image_cursor']!=12*s for s in range(1,20001)):
            raise ValueError('Incomplete step/cursor log')
        results.append(read(d/'val_step020000.json'))
    if any(logs[0][s]['data_ids_sha256']!=logs[1][s]['data_ids_sha256'] for s in range(1,20001)):
        raise ValueError('Training images differ')
    baseline=read(dirs[0]/'baseline/val_step018000.json')
    for r in [*results,baseline]:
        if r['samples']!=14249: raise ValueError('Require full validation')
    for label,r in zip(('frame','fixed','baseline'),[*results,baseline]):
        for mode in ('ssi','none'): print(label,mode,r['metrics'][mode])
    f=results[0]; conditions={}
    for label,reference in (('vs_fixed',results[1]),('vs_baseline',baseline)):
        a,b=f['metrics']['ssi'],reference['metrics']['ssi']
        conditions[label]=dict(absrel_gain=b['abs_rel']-a['abs_rel']>=.0005,
            rmse_guard=a['rmse']<=1.005*b['rmse'],a1_guard=a['a1']>=b['a1']-.001,
            sequences=sum(f['sequences']['ssi'][s]['abs_rel']<v['abs_rel'] for s,v in reference['sequences']['ssi'].items())>=2)
    print(json.dumps(conditions,indent=2))
    print('Predeclared investment criteria, not a significance test; no further jobs submitted.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True); p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    pair_smoke(args.base) if args.smoke else report(args.base)
