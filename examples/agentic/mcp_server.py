# TurboQuant Pro -- Model Context Protocol (MCP) server
# MIT License
"""Serve TurboQuant Pro as an MCP tool server.

This is the "wrap it as an AI skill" path: any MCP-capable client (Claude
Desktop, an IDE agent, a custom host) discovers and calls these tools without
writing code. Each tool delegates to :mod:`turboquant_pro.agent_tools`.

The tools take the *goal* as an explicit argument -- recall ``k``/``min_recall``
or the ``consumer`` metric -- because "keep the angle" means the goal is
dynamic: the calling model picks the acceptance target for the task at hand, and
the answer is accepted/certified on that goal's retrieval fidelity, never on
reconstruction cosine.

Run:
    pip install turboquant-pro "mcp[cli]" numpy
    python mcp_server.py                 # stdio transport

Register with an MCP host (e.g. Claude Desktop config):
    {
      "mcpServers": {
        "turboquant-pro": {
          "command": "python",
          "args": ["/abs/path/to/examples/agentic/mcp_server.py"]
        }
      }
    }

The flagship flow the user asks for -- "what's the best compression ratio at
0.99 recall for this corpus?" -- is `best_compression_at_recall`, and the
mathematical receipt is `certify_ranking`.
"""

from __future__ import annotations

import numpy as np
from mcp.server.fastmcp import FastMCP

from turboquant_pro import (
    best_compression_at_recall,
    certify_ranking,
    recommend_kv_key_quantizer,
)

mcp = FastMCP("turboquant-pro")


@mcp.tool()
def best_compression_at_recall_tool(
    corpus_npy: str, k: int = 10, min_recall: float = 0.99
) -> dict:
    """Best compression ratio for a corpus that keeps recall@k >= min_recall.

    Args:
        corpus_npy: Path to a .npy file holding a 2-D embedding matrix (n, dim).
        k: Retrieval cut-off (top-k neighbours). The goal is dynamic -- set it.
        min_recall: Recall threshold in [0, 1] to hold. Accepted on recall, not
            cosine.
    """
    return best_compression_at_recall(np.load(corpus_npy), k=k, min_recall=min_recall)


@mcp.tool()
def certify_ranking_tool(
    original_npy: str, reconstructed_npy: str, metric: str = "cosine"
) -> dict:
    """Distribution-free guaranteed floor on rank agreement (the certificate).

    Args:
        original_npy: Path to .npy of the reference vectors (n, dim).
        reconstructed_npy: Path to .npy of the decompressed vectors (n, dim).
        metric: Ranking metric to certify ("cosine" or "l2").
    """
    return certify_ranking(
        np.load(original_npy), np.load(reconstructed_npy), metric=metric
    )


@mcp.tool()
def recommend_kv_key_quantizer_tool(
    keys_npy: str, consumer: str = "attention_logits", bits: int = 4
) -> dict:
    """(A2) probe: recommend polar vs. per-channel quantization for attention keys.

    Args:
        keys_npy: Path to .npy of an attention-key sample (n, dim).
        consumer: Downstream metric to preserve (the dynamic goal).
        bits: Probe bit budget per dimension.
    """
    return recommend_kv_key_quantizer(np.load(keys_npy), consumer=consumer, bits=bits)


if __name__ == "__main__":
    mcp.run()
