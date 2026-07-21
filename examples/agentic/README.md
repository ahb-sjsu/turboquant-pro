# TurboQuant Pro for agents (LangChain · DSPy · MCP · custom GPT)

This directory makes TurboQuant Pro **directly consumable by autonomous
systems** — an agent can discover the tools, call them, and get back a JSON
decision plus a mathematical certificate, without a human in the loop.

Everything here is a thin wrapper over one importable module,
[`turboquant_pro.agent_tools`](../../turboquant_pro/agent_tools.py), which is the
single source of truth for the tool contract. The frameworks below are just
transport.

## The design principle: the goal is dynamic

"Keep the angle" says the coordinate worth preserving is whichever one carries
the **task's** geometry — and which coordinate that is depends on the goal. So
the goal is a **runtime input, not a library constant**. The agent declares it
per call:

- *which* `k` and `min_recall` (for retrieval), or
- *which* `consumer` metric (`attention_logits`, `cosine`, `l2`) for KV keys.

The library then reports the best compression **for that goal**, certified
**against that goal**. This is exactly why reconstruction **cosine is never the
acceptance gate** — cosine is a fixed, goal-agnostic proxy, blind to the ranking
the current task actually consumes. Acceptance is always the declared goal's
retrieval fidelity (`recall@k`), the consumer-metric (A2) probe, or the
distribution-free rank certificate. Cosine is reported only as a labelled
diagnostic.

## The tools

| Tool | Answers | Accepts on |
|------|---------|-----------|
| `best_compression_at_recall` | "How small can this corpus get while keeping recall@k?" | **recall@k** |
| `recommend_kv_key_quantizer` | "Polar or per-channel for these attention keys?" | **(A2) consumer-metric ranking** |
| `certify_ranking` | "What rank fidelity can I *guarantee*?" | **distribution-free floor** |

The flagship flow — *"what's the best compression ratio at 0.99 recall for this
corpus?"* — is `best_compression_at_recall`, and the receipt is
`certify_ranking`.

## Zero-framework (just the library)

```python
import numpy as np
from turboquant_pro import best_compression_at_recall, certify_ranking

corpus = np.load("corpus.npy")                     # (n, dim) float32
plan = best_compression_at_recall(corpus, k=10, min_recall=0.99)
print(plan["compression_ratio"], plan["meets_target"])   # e.g. 21.3, True
# plan is JSON-serializable -> hand straight back to a tool-calling model
```

## LangChain

```bash
pip install turboquant-pro langchain-core numpy
```

```python
from langchain_tools import TQP_TOOLS          # ./langchain_tools.py
agent = create_react_agent(llm, TQP_TOOLS, prompt)
```

Each `@tool` takes `.npy` paths so a model can pass a large corpus by reference.

## DSPy

```bash
pip install turboquant-pro dspy-ai numpy
```

```python
import dspy
from dspy_tools import TQP_TOOLS, CompressionAdvisor   # ./dspy_tools.py
advisor = CompressionAdvisor()
advisor(question="Best compression for corpus.npy keeping recall@10 >= 0.99?")
```

## MCP (the "AI skill" path)

```bash
pip install turboquant-pro "mcp[cli]" numpy
python mcp_server.py            # ./mcp_server.py, stdio transport
```

Register with any MCP host (e.g. Claude Desktop):

```json
{
  "mcpServers": {
    "turboquant-pro": {
      "command": "python",
      "args": ["/abs/path/to/examples/agentic/mcp_server.py"]
    }
  }
}
```

## Custom GPT / OpenAI Actions / tool indexing

[`tool_manifest.json`](./tool_manifest.json) holds JSON-Schema definitions for
the three tools — register them with any tool-calling API or generate an Actions
spec from it. It is produced by `turboquant_pro.agent_tools.list_tools()`, so it
stays in sync with the code:

```python
import json
from turboquant_pro import list_tools
print(json.dumps(list_tools(), indent=2))
```

The intended hook: a user uploads a `.npy` corpus and asks *"what's the best
compression ratio I can get while keeping 0.99 recall?"*. The GPT calls
`best_compression_at_recall`, then `certify_ranking`, and prints the ratio with
its guaranteed rank floor — a profound geometric result reduced to one API call.
