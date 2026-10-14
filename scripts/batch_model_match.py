#!/usr/bin/env python3
"""Batch model selection of the best current Intent for each raw Action."""
from __future__ import annotations
import argparse, json, os, concurrent.futures, urllib.request
from pathlib import Path

RULES = Path('docs/9.28intent_action_matching_rules_v1.md')
LABELS = {'exact_match','partial_match','object_mismatch','type_mismatch','no_intent_match','ambiguous'}

def call(base,key,model,payload,timeout):
    body={'model':model,'temperature':0,'messages':[{'role':'system','content':RULES.read_text(encoding='utf8')+'\nReturn JSON only.'},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]}
    req=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req,timeout=timeout) as r: return json.loads(r.read())['choices'][0]['message']['content']

def one(item,args):
    if not item['intents']:
        return {**item['base'],'selected_intent_ids':[],'label':'no_intent_match','confidence':1.0,'reason':'No due_now Intent exists for this step.','alternative_intent_ids':[],'review_required':False}
    for attempt in range(args.retries):
        try:
            raw=call(args.base_url,args.api_key,args.model,{'action':item['action'],'intents':item['intents']},args.timeout)
            raw=raw.strip().removeprefix('```json').removesuffix('```').strip(); out=json.loads(raw)
            if out.get('label') not in LABELS: raise ValueError('invalid label')
            if 'selected_intent_ids' not in out:
                legacy = out.get('selected_intent_id')
                out['selected_intent_ids'] = [] if legacy in (None, '') else [legacy]
            out['selected_intent_ids'] = [x for x in out['selected_intent_ids'] if isinstance(x, str)]
            out.setdefault('alternative_intent_ids',[]); out.setdefault('confidence',0.0); out.setdefault('reason','')
            out['review_required']=bool(out['review_required'] if 'review_required' in out else out['label'] in {'ambiguous','partial_match','no_intent_match'} or item['action'].get('requires_manual_semantic_review') or len(item['intents'])>1 or out['confidence']<.8)
            return {**item['base'],**out}
        except Exception as e:
            err=str(e)
    return {**item['base'],'selected_intent_ids':[],'label':'ambiguous','confidence':0.0,'reason':'model_failed:'+err,'alternative_intent_ids':[],'review_required':True}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--intents',type=Path,required=True); ap.add_argument('--actions',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--base-url',required=True); ap.add_argument('--model',required=True); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--timeout',type=int,default=120); ap.add_argument('--retries',type=int,default=3); args=ap.parse_args(); args.api_key=os.environ['INTENT_API_KEY']
    intents={}
    for l in args.intents.open(encoding='utf8'):
        r=json.loads(l); intents[(r['trajectory_id'],r['thought_step'])]=[i for i in r.get('execution_intents',[]) if i.get('match_eligibility')=='due_now']
    items=[]
    for l in args.actions.open(encoding='utf8'):
        a=json.loads(l); key=(a['trajectory_id'],a['thought_step']); items.append({'action':a,'intents':intents.get(key,[]),'base':{'trajectory_id':a['trajectory_id'],'thought_step':a['thought_step'],'action_id':a['action_id'],'atomicity':a['atomicity'],'raw_action_sha256':a['raw_action_sha256']}})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    existing={}
    if args.output.exists():
        for l in args.output.open(encoding='utf8'):
            r=json.loads(l); existing[r['action_id']]=r
    todo=[x for x in items if x['base']['action_id'] not in existing]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(lambda x: one(x,args),todo): existing[r['action_id']]=r
    with args.output.open('w') as h:
        for k in sorted(existing): h.write(json.dumps(existing[k],ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps({'actions':len(items),'written':len(existing),'pending':len(todo)},ensure_ascii=False))
if __name__=='__main__': main()
