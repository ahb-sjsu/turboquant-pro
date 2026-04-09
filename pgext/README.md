# tqvector — TurboQuant Compressed Vector Type for PostgreSQL

A PostgreSQL extension written in Rust (via [pgrx](https://github.com/pgcentralfoundation/pgrx)) that adds the `tqvector` data type for storing and searching compressed embeddings.

## Features

- **`tqvector` type** — stores embeddings as bit-packed scalar-quantized vectors
- **2/3/4-bit quantization** — 5-10x compression vs float32
- **`<=>` operator** — cosine distance for ORDER BY queries
- **Zero Python dependency** — pure Rust, runs inside PostgreSQL

## Quick Start

```sql
-- Compress an embedding
SELECT tq_compress(ARRAY[0.1, 0.2, 0.3, ...]::float4[], 3);
-- Returns: tqvector(1024-dim, 3-bit, norm=1.2345, 10.5x)

-- Store compressed vectors
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    tqv tqvector
);

INSERT INTO embeddings (content, tqv)
SELECT content, tq_compress(embedding, 3)
FROM raw_embeddings;

-- Search by cosine similarity
SELECT id, content, tq_cosine_sim(tqv, tq_compress(query_embedding, 3)) AS sim
FROM embeddings
ORDER BY tqv <=> tq_compress(query_embedding, 3)
LIMIT 10;

-- Inspect compression
SELECT tq_dim(tqv), tq_bits(tqv), tq_ratio(tqv), tq_norm(tqv)
FROM embeddings LIMIT 1;
-- 1024, 3, 10.5, 1.2345
```

## Build & Install

```bash
# Install pgrx
cargo install cargo-pgrx
cargo pgrx init

# Build and install
cd pgext
cargo pgrx install --release

# Enable in your database
psql -c "CREATE EXTENSION tqvector;"
```

## Functions

| Function | Description |
|----------|-------------|
| `tq_compress(float4[], int)` | Compress embedding to tqvector |
| `tq_decompress(tqvector)` | Decompress back to float4[] |
| `tq_cosine_sim(tqvector, tqvector)` | Cosine similarity (0-1) |
| `tq_cosine_dist(tqvector, tqvector)` | Cosine distance (1-sim) |
| `tq_dim(tqvector)` | Get embedding dimension |
| `tq_bits(tqvector)` | Get quantization bits |
| `tq_ratio(tqvector)` | Get compression ratio |
| `tq_norm(tqvector)` | Get original L2 norm |
| `tq_size_bytes(tqvector)` | Get compressed size |
| `<=>` operator | Cosine distance for ORDER BY |

## Compression Quality

| Bits | Ratio | Cosine Sim | Recall@10 |
|------|-------|-----------|-----------|
| 2 | ~15x | 0.924 | 78.7% |
| 3 | ~10x | 0.978 | 83.8% |
| 4 | ~8x | 0.995 | 90.4% |

## License

MIT
