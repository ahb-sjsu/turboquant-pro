# Integrations

How to use TurboQuant Pro compression with pgvector, a native PostgreSQL extension, NATS transport, serving frameworks, and cross-framework vector-database exports.

Part of [TurboQuant Pro](../README.md).

#### pgvector (PostgreSQL, Python)
Compress high-dimensional embeddings stored in pgvector — 10× from float32, 5× from float16:

```python
from turboquant_pro import TurboQuantPGVector

tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)
compressed = tq.compress_embedding(embedding_float32)   # 4096 -> 388 bytes
bytea_data = compressed.to_pgbytea()                    # store as bytea
compressed_batch = tq.compress_batch(embeddings_array)
scores = tq.compressed_cosine_similarity(query, compressed_batch)

tq.create_compressed_table(conn, "embeddings_compressed")
tq.insert_compressed(conn, "embeddings_compressed", ids, embeddings)
results = tq.search_compressed(conn, "embeddings_compressed", query, top_k=10)
```

#### Native PostgreSQL extension (Rust + CUDA)
`pgext/` is a native extension (Rust/pgrx) adding the `tqvector` type directly to PostgreSQL — no Python needed. **194K BGE-M3 vectors: 23,969 vec/sec, 31× storage reduction, 12 Rust unit tests passing.**

```sql
CREATE TABLE embeddings_tq AS
SELECT id, tq_compress(embedding::float4[], 3) AS tqv FROM embeddings;

SELECT id, tqv <=> tq_compress(query::float4[], 3) AS dist
FROM embeddings_tq ORDER BY dist LIMIT 10;
```
```bash
cd pgext && cargo install cargo-pgrx && cargo pgrx init --pg16 $(which pg_config)
cargo pgrx install --release && psql -c "CREATE EXTENSION tqvector;"
```
Optional GPU: `cargo build --features gpu` (CUDA 12.0+, cudarc). See [`pgext/README.md`](../pgext/README.md).

#### NATS transport codec
Compress embeddings for NATS JetStream or any message bus (4096 → 392 bytes):

```python
from turboquant_pro import TurboQuantNATSCodec

codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)
payload = codec.encode(embedding_float32)        # 4096 -> 392 bytes
embedding_approx = codec.decode(payload)
print(codec.stats())  # {'compression_ratio': 10.45, ...}
```
In production, [nats-bursting](https://github.com/ahb-sjsu/nats-bursting) uses this on the burst path to a shared Kubernetes cluster: over a real ~2 Mbps hop the ~10× smaller payload cut NATS round-trip **up to 8.4× at 256 KB**, fidelity distribution-agnostic (real bge-small reproduces the random-vector ratio/cosine within 0.001).

#### vLLM, HuggingFace, llama.cpp
```python
from turboquant_pro.vllm_plugin import TurboQuantKVManager

mgr = TurboQuantKVManager(n_layers=32, n_kv_heads=8, head_dim=128, bits=3, hot_window=512)
mgr.store(layer_id=0, keys=k_tensor, values=v_tensor)
keys, values = mgr.load(layer_id=0, start=0, end=1024)   # decompresses cold storage
max_ctx = mgr.estimate_capacity(max_memory_gb=4.0)        # ~32K instead of ~8K
```
- **HuggingFace Transformers:** wrap the KV cache in `generate()` by subclassing the attention layer (`tq.compress(key_states, packed=True)` on update, decompress when scoring).
- **llama.cpp / llama-cpp-python:** see `examples/llama_integration.py` for the KV-intercept pattern.

#### Cross-framework export
`export_compressed(ids, embeddings, tq, format="qdrant")` formats compressed embeddings for **Milvus, Qdrant, Weaviate, Pinecone**, or a portable JSON format (decompressed float for native search + compressed bytes as base64 for storage).
