#!/usr/bin/env python3
"""Generate deterministic Stage 2 Intent--Action candidates and preliminary labels."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

def norm(s):
    return set(re.findall(r"[a-z0-9_./-]+", s.lower()))
def action_text(a):
    fn=a.get("function",""); args=a.get("args",{})
    return fn+" "+json.dumps(args,ensure_ascii=False,sort_keys=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--intents',type=Path,required=True); ap.add_argument('--actions',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    acts={}
    with args.actions.open() as h:
      for l in h:
        if l.strip():
          r=json.loads(l); acts.setdefault((r['trajectory_id'],r['thought_step']),[]).append(r)
    out=[]; counts={}
    with args.intents.open() as h:
      for l in h:
        if not l.strip(): continue
        row=json.loads(l); key=(row['trajectory_id'],row['thought_step'])
        for it in row.get('execution_intents',[]):
          if it.get('match_eligibility')!='due_now': continue
          candidates=acts.get(key,[])
          if not candidates:
            base={**it,'trajectory_id':row['trajectory_id'],'thought_step':row['thought_step'],'pair_id':hashlib.sha256((it['intent_id']+'|no_action').encode()).hexdigest()[:16],'action':None,'preliminary_label':'no_action','candidate_basis':'absent','risk_level':'high','risk_reasons':['no_action']}
            out.append(base); counts['no_action']=counts.get('no_action',0)+1; continue
          itoks=norm(it['intent_text']+' '+it.get('source_quote',''))
          for a in candidates:
            atoks=norm(action_text(a['action'])); overlap=len(itoks&atoks)
            label='exact_match' if overlap>=2 else ('partial_match' if overlap==1 else 'type_mismatch')
            pair={**it,'trajectory_id':row['trajectory_id'],'thought_step':row['thought_step'],'pair_id':hashlib.sha256((it['intent_id']+'|'+a['action_id']).encode()).hexdigest()[:16],'action':a['action'],'action_id':a['action_id'],'atomicity':a['atomicity'],'action_has_extras':a['action_has_extras'],'preliminary_label':label,'candidate_basis':'both' if overlap else 'operation','risk_level':'high' if len(candidates)>1 or a['action_has_extras'] else 'low','risk_reasons':['multiple_candidates'] if len(candidates)>1 else (['action_has_extras'] if a['action_has_extras'] else [])}
            out.append(pair); counts[label]=counts.get(label,0)+1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as h:
      for r in out: h.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps({'pair_count':len(out),'label_counts':counts},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
