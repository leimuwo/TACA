#!/usr/bin/env python3
"""Use GQQ to identify Intent records that are ambiguous without hidden context.

The script is resumable. It reads an input JSONL prepared from the BGE split,
submits small JSON batches, and writes one decision per Intent. It never edits
the source train/dev/test files.
"""
from __future__ import annotations

import argparse, concurrent.futures, json, os, time, urllib.request
from pathlib import Path

SYSTEM = """You screen tool-calling Intent texts for dataset quality. An Intent is ambiguous when, using only the Intent text (not hidden trajectory state), a reviewer cannot determine the target object, operation, scope, or required result well enough to choose the correct Action from plausible candidates. Mark ambiguous for vague references (the file, remaining items, all outputs), missing target/topic, underspecified multi-object work, or when multiple candidate Actions are equally valid. Do not mark ambiguous merely because the Action command is long or because several implementation details are possible if the Intent clearly specifies the goal and target. Return JSON only."""

def compact(x): return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def call(endpoint, key, model, rows, timeout):
    payload = {"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": SYSTEM + "\nSchema: JSON array; each item has query_id, ambiguous (boolean), confidence (0..1), category (one of clear_target, missing_target, vague_reference, underspecified_scope, competing_valid_actions, unclear_operation), reason (short), missing_information (array)."},
        {"role": "user", "content": compact({"items": rows})},
    ]}
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        data = json.loads(response.read())
    text = data["choices"][0]["message"]["content"].strip().removeprefix("```json").removesuffix("```").strip()
    out = json.loads(text)
    if not isinstance(out, list): raise ValueError("GQQ response is not an array")
    return out

def load_env(path):
    if not path.exists(): return
    for line in path.read_text(encoding="utf8").splitlines():
        line=line.strip()
        if line and not line.startswith("#") and "=" in line:
            k,v=line.split("=",1); os.environ.setdefault(k.strip().removeprefix("export "), v.strip().strip("\"'"))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--env",type=Path); ap.add_argument("--endpoint",default=None); ap.add_argument("--model",default=None); ap.add_argument("--workers",type=int,default=2); ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--retries",type=int,default=5); ap.add_argument("--timeout",type=int,default=180); args=ap.parse_args()
    load_env(args.env) if args.env else None
    key=os.environ.get("INF_API_KEY") or os.environ.get("INTENT_API_KEY")
    endpoint=args.endpoint or os.environ.get("INF_API_ENDPOINT") or os.environ.get("INTENT_API_BASE_URL") or "https://qjkpcombh9jkcecah8jdkgd5d5beqege.openapi-qb-ai.sii.edu.cn"
    model=args.model or os.environ.get("INF_MODEL_NAME") or os.environ.get("INTENT_MODEL_NAME") or "deepseek-v4-flash-gqq"
    if not key: raise SystemExit("missing INF_API_KEY/INTENT_API_KEY")
    if not endpoint: raise SystemExit("missing INF_API_ENDPOINT/INTENT_API_BASE_URL")
    rows=[json.loads(x) for x in args.input.read_text(encoding="utf8").splitlines() if x.strip()]
    done={}
    if args.output.exists():
        for x in args.output.read_text(encoding="utf8").splitlines():
            if x.strip():
                y=json.loads(x); done[y["query_id"]]=y
    todo=[x for x in rows if x["query_id"] not in done]
    batches=[todo[i:i+args.batch_size] for i in range(0,len(todo),args.batch_size)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def one(batch):
        last=""
        for attempt in range(args.retries):
            try:
                result=call(endpoint,key,model,batch,args.timeout)
                if {x.get("query_id") for x in result}!={x["query_id"] for x in batch}: raise ValueError("response query IDs do not match batch")
                for x in result:
                    if not isinstance(x.get("ambiguous"),bool) or not isinstance(x.get("confidence"),(int,float)) or not isinstance(x.get("reason"),str): raise ValueError("invalid decision schema")
                return result
            except Exception as exc: last=str(exc); time.sleep(min(2**attempt,20))
        return [{"query_id":x["query_id"],"ambiguous":None,"confidence":0,"category":"screen_failed","reason":last,"missing_information":[]} for x in batch]
    with args.output.open("a",encoding="utf8") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(one,batches):
            for x in result: stream.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
            stream.flush()
    print(json.dumps({"input":len(rows),"existing":len(done),"processed":len(todo),"output":len(rows)},ensure_ascii=False))

if __name__=="__main__": main()
