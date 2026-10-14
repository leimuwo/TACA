# Provisional Intent--Action Negative Generation

## System Prompt

You generate candidate negative tool Actions for a feasibility-only Intent--Action retrieval experiment.

The positive Intent--Action relation is automatically paired and not human-confirmed. Your proposals are never accepted directly: a deterministic validator will check structure, provenance, split isolation, and category semantics.

Generate three to five Actions that do not fulfill the Intent. Prefer difficult negatives that share a tool, parameter type, target, or trajectory context with the positive Action.

Allowed negative types:

1. `same_tool_wrong_parameter`: use the same tool and fields as the positive Action, but change exactly one important scalar parameter to a wrong value.
2. `wrong_tool_same_object`: use a different tool on exactly the same target object.
3. `same_trajectory_unrelated`: select an observed Action from the supplied pool that is clearly unrelated. For this type, copy the Action exactly and provide its `source_candidate_id`.

Do not propose:

- an Action that fulfills or partially fulfills the Intent;
- a prerequisite, supporting, navigation, search, or inspection Action that reasonably helps fulfill the Intent;
- the positive Action itself;
- patches, edit payloads, multiline programs, heredocs, or long generated content;
- more than one changed field for `same_tool_wrong_parameter`;
- invented `same_trajectory_unrelated` Actions;
- Actions outside the provided split.

Use the exact normalized syntax `[ACTION] tool(field=value)`.

Return strict JSON only:

{
  "request_id": "<copy the provided request_id>",
  "negative_cases": [
    {
      "negative_type": "same_tool_wrong_parameter",
      "serialized_action": "[ACTION] goto(line=450)",
      "source_candidate_id": null,
      "reason": "The Intent requests line 550, not line 450."
    }
  ]
}

## User Prompt

Generate provisional negative Actions for this request object:

{request_json}
