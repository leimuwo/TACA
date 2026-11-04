"""Standalone extraction validation and legacy Action serialization."""

import json

SCOPES = {'current', 'future'}

def parse_prediction(raw,thought):
 try:value=json.loads(raw)
 except (ValueError,TypeError):return None,'invalid_json'
 if not isinstance(value,dict) or set(value)!={'intents'} or not isinstance(value['intents'],list):return None,'invalid_schema'
 seen=set();positions=[]
 for i in value['intents']:
  if not isinstance(i,dict) or set(i)!={'intent_quote','temporal_scope'}:return None,'invalid_schema'
  if not isinstance(i['intent_quote'],str) or not i['intent_quote'].strip() or i['temporal_scope'] not in SCOPES:return None,'invalid_scope_or_quote'
  if i['intent_quote'] not in thought:return None,'non_source_quote'
  if i['intent_quote'] in seen:return None,'duplicate_quote'
  seen.add(i['intent_quote']);positions.append(thought.find(i['intent_quote']))
 if positions!=sorted(positions):return None,'source_order_wrong'
 return value['intents'],'valid'

TARGET_KEYS = ("path", "file", "url", "uri", "target", "command", "cmd", "method")

def prioritize_action(text):
    """Move tool and target fields ahead of long payloads before pair truncation."""
    prefix = "[ACTION] "
    if not text.startswith(prefix):
        raise ValueError("invalid action serialization")
    try:
        action = json.loads(text[len(prefix):])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid action JSON") from exc
    if not isinstance(action, dict):
        raise ValueError("action must be a JSON object")
    args = action.get("args")
    if isinstance(args, dict):
        ordered_args = {key: args[key] for key in TARGET_KEYS if key in args}
        ordered_args.update({key: value for key, value in args.items() if key not in ordered_args})
        action = {"function": action.get("function"), "args": ordered_args,
                  **{key: value for key, value in action.items() if key not in ("function", "args")}}
    return prefix + json.dumps(action, ensure_ascii=False, separators=(",", ":"))
