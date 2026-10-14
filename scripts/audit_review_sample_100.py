#!/usr/bin/env python3
"""Audit common data issues in the representative 100 review cases."""
from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sample',type=Path,required=True); ap.add_argument('--intents',type=Path,required=True); ap.add_argument('--actions',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=[json.loads(l) for l in args.sample.open(encoding='utf8') if l.strip()]
    steps={(r['trajectory_id'],r['thought_step']):r for r in map(json.loads,args.intents.open(encoding='utf8'))}
    acts={r['action_id']:r for r in map(json.loads,args.actions.open(encoding='utf8'))}
    counts=Counter(); examples=defaultdict(list); findings=[]
    for r in rows:
      step=steps[(r['trajectory_id'],r['thought_step'])]; intents=[i for i in step.get('execution_intents',[]) if i.get('match_eligibility')=='due_now']; a=acts[r['action_id']]; raw=a['action']; local=[]
      sig=[(i.get('intent_text','').strip().lower(),i.get('source_quote','').strip().lower()) for i in intents]
      if len(sig)!=len(set(sig)) or len({i.get('source_quote') for i in intents})<len(intents): local.append('intent_duplicate_or_same_quote')
      if len(intents)>1: local.append('multi_intent_step')
      if r['label'] in {'partial_match','object_mismatch','type_mismatch'}: local.append('match_needs_manual_check')
      if r['label']=='no_intent_match': local.append('no_intent_match')
      if r['label']=='ambiguous': local.append('ambiguous')
      if r.get('confidence',1)<.8: local.append('low_confidence')
      if raw.get('function')=='exec' and isinstance(raw.get('args'),dict) and isinstance(raw['args'].get('command'),str):
        cmd=raw['args']['command']
        if '\n' in cmd: local.append('action_multiline')
        if len(cmd)>1000: local.append('action_very_long_command')
        if ('&&' in cmd or ';' in cmd) and re.search(r'(^|\s)(cd|mkdir|sha256sum|wc|echo|cat)(\s|$)',cmd): local.append('action_composite_extra_ops')
        if 'python' in cmd and ('open(' in cmd or 'write(' in cmd) and len(cmd)>500: local.append('action_embedded_script')
      if local:
        for f in set(local): counts[f]+=1; examples[f].append({'trajectory_id':r['trajectory_id'],'thought_step':r['thought_step'],'action_id':r['action_id'],'label':r['label']})
        findings.append({'action_id':r['action_id'],'flags':sorted(set(local))})
    report={'sample_count':len(rows),'counts':dict(counts),'examples':{k:v[:10] for k,v in examples.items()},'findings':findings,'note':'规则审计用于定位常见问题，未替代人工最终判定。'}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'sample_count':len(rows),'counts':dict(counts)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
