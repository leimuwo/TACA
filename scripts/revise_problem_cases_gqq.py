#!/usr/bin/env python3
"""Use GQQ to revise flagged Intent–Action cases with deduplication and coverage."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, urllib.request, threading
from pathlib import Path

RULES = Path('docs/9.28intent_action_matching_rules_v1.md')

def request(args, payload):
    body={'model':args.model,'temperature':0,'messages':[{'role':'system','content':RULES.read_text(encoding='utf8')+'\nYou are revising a flagged dataset case. Return JSON only.'},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]}
    req=urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Authorization':'Bearer '+args.key,'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req,timeout=args.timeout) as response:
        return json.loads(response.read())['choices'][0]['message']['content']

def revise(item,args):
    payload={'case':item,'task':'Review this case. Deduplicate equivalent due_now Intents, split independent targets, modify Intent text only when unsupported by the Thought, then assign this raw Action zero or more revised Intent IDs. Return JSON object with revised_intents (array of objects with intent_id,intent_text,source_quote,temporal_scope,revision_type,parent_intent_id), selected_intent_ids (array), label (one of exact_match,partial_match,object_mismatch,type_mismatch,no_intent_match,ambiguous), confidence (0..1), reason, revision_notes. Keep each original Intent unless there is clear evidence to revise it.'}
    last=''
    for attempt in range(args.retries):
        try:
            raw=request(args,payload).strip().removeprefix('```json').removesuffix('```').strip(); out=json.loads(raw)
            if not isinstance(out,dict) or not isinstance(out.get('revised_intents'),list) or not isinstance(out.get('selected_intent_ids'),list) or out.get('label') not in {'exact_match','partial_match','object_mismatch','type_mismatch','no_intent_match','ambiguous'}: raise ValueError('invalid revision schema')
            allowed={i['intent_id'] for i in out['revised_intents'] if isinstance(i,dict) and isinstance(i.get('intent_id'),str)}
            if not set(out['selected_intent_ids'])<=allowed: raise ValueError('selected Intent ID absent from revised_intents')
            if any(not isinstance(i,dict) or not isinstance(i.get('source_quote'),str) or i['source_quote'] not in item['thought'] for i in out['revised_intents']): raise ValueError('invalid source_quote')
            out['revision_status']='gqq_revised'; out['original_action_id']=item['action_id']; return out
        except Exception as e: last=str(e)
    return {'revision_status':'revision_failed','original_action_id':item['action_id'],'revision_error':last}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--base-url',required=True); ap.add_argument('--model',default='deepseek-v4-flash-gqq'); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--timeout',type=int,default=180); ap.add_argument('--retries',type=int,default=8); args=ap.parse_args(); args.key=os.environ['INTENT_API_KEY']
    rows=[json.loads(l) for l in args.input.open(encoding='utf8') if l.strip()]; args.output.parent.mkdir(parents=True,exist_ok=True)
    existing={}
    if args.output.exists():
      for l in args.output.open(encoding='utf8'):
       if l.strip(): existing[json.loads(l)['original_action_id']]=json.loads(l)
    todo=[r for r in rows if r['action_id'] not in existing]
    lock=threading.Lock()
    with args.output.open('a',encoding='utf8') as out, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
      futures=[ex.submit(revise,x,args) for x in todo]
      for future in concurrent.futures.as_completed(futures):
        result=future.result()
        with lock:
          out.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+'\n');out.flush();os.fsync(out.fileno())
          existing[result['original_action_id']]=result
    print(json.dumps({'input':len(rows),'written':len(existing),'processed_now':len(todo)},ensure_ascii=False))
if __name__=='__main__': main()
