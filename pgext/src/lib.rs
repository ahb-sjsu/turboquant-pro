//! TurboQuant compressed vector type for PostgreSQL.
//!
//! Provides `tqvector` — a compressed embedding type that stores vectors
//! as bit-packed scalar-quantized indices with a separate L2 norm.
//! Supports 2, 3, and 4-bit quantization via PolarQuant + Lloyd-Max.
//!
//! # Usage
//! ```sql
//! -- Single vector
//! SELECT tq_compress(embedding, 3) FROM chunks LIMIT 1;
//!
//! -- Bulk compress a whole table (GPU-accelerated if available)
//! SELECT tq_bulk_compress('chunks', 'embedding', 'chunks_tq', 3);
//!
//! -- Search with cosine distance
//! SELECT id, tqv <=> query_tqv AS dist
//! FROM chunks_tq ORDER BY dist LIMIT 10;
//! ```
//!
//! Copyright (c) 2026 Andrew H. Bond. MIT License.

use pgrx::prelude::*;

pgrx::pg_module_magic!();

mod codebook;
mod compress;
mod gpu;
mod types;

pub use types::TqVector;

// ─── Single-vector functions ─────────────────────────────────────

/// Compress a float4 array to a tqvector.
#[pg_extern(immutable, parallel_safe)]
fn tq_compress(embedding: Vec<f32>, bits: default!(i32, 3)) -> TqVector {
    if bits != 2 && bits != 3 && bits != 4 {
        pgrx::error!("bits must be 2, 3, or 4");
    }
    compress::compress(&embedding, bits as u8, 42)
}

/// Decompress a tqvector back to a float4 array.
#[pg_extern(immutable, parallel_safe)]
fn tq_decompress(tqv: TqVector) -> Vec<f32> {
    compress::decompress(&tqv)
}

/// Cosine similarity between two compressed vectors.
#[pg_extern(immutable, parallel_safe)]
fn tq_cosine_sim(a: TqVector, b: TqVector) -> f32 {
    if a.dim != b.dim {
        pgrx::error!("dimension mismatch: {} vs {}", a.dim, b.dim);
    }
    let va = compress::decompress(&a);
    let vb = compress::decompress(&b);

    let dot: f32 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = va.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = vb.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-30 || norm_b < 1e-30 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Cosine distance (1 - similarity) for ORDER BY.
#[pg_extern(immutable, parallel_safe)]
fn tq_cosine_dist(a: TqVector, b: TqVector) -> f32 {
    1.0 - tq_cosine_sim(a, b)
}

// ─── Metadata functions ──────────────────────────────────────────

#[pg_extern(immutable, parallel_safe)]
fn tq_dim(tqv: TqVector) -> i32 { tqv.dim as i32 }

#[pg_extern(immutable, parallel_safe)]
fn tq_bits(tqv: TqVector) -> i32 { tqv.bits as i32 }

#[pg_extern(immutable, parallel_safe)]
fn tq_ratio(tqv: TqVector) -> f32 {
    let original = tqv.dim as f32 * 4.0;
    let compressed = tqv.data.len() as f32 + 4.0;
    original / compressed
}

#[pg_extern(immutable, parallel_safe)]
fn tq_norm(tqv: TqVector) -> f32 { tqv.norm }

#[pg_extern(immutable, parallel_safe)]
fn tq_size_bytes(tqv: TqVector) -> i32 {
    (tqv.data.len() + 4) as i32
}

// ─── Batch compress (CPU, amortized rotation) ────────────────────

// Note: tq_compress_batch for arrays of arrays is not practical
// in SQL. Use tq_bulk_compress() for table-level batch compression.

// ─── Bulk compress utility ───────────────────────────────────────

/// Bulk compress all embeddings from a source table into a target table.
///
/// Reads vectors in batches, compresses with amortized rotation
/// (one rotation matrix per batch, not per vector), and inserts
/// into the target table.
///
/// Returns the number of vectors compressed.
///
/// Usage:
///   SELECT tq_bulk_compress('chunks', 'embedding', 'chunks_tq', 3, 500);
#[pg_extern]
fn tq_bulk_compress(
    source_table: &str,
    source_column: &str,
    target_table: &str,
    bits: default!(i32, 3),
    batch_size: default!(i32, 500),
) -> i64 {
    if bits != 2 && bits != 3 && bits != 4 {
        pgrx::error!("bits must be 2, 3, or 4");
    }

    let bits_u8 = bits as u8;
    let batch = batch_size as usize;

    // Create target table
    Spi::run(&format!(
        "CREATE TABLE IF NOT EXISTS {} (id TEXT, tqv tqvector)",
        target_table
    )).unwrap_or_else(|e| pgrx::error!("Create table: {}", e));

    // Count
    let total: i64 = Spi::get_one(&format!(
        "SELECT COUNT(*)::int8 FROM {} WHERE {} IS NOT NULL",
        source_table, source_column
    )).unwrap_or(Some(0)).unwrap_or(0);

    if total == 0 {
        pgrx::notice!("No vectors found");
        return 0;
    }

    pgrx::notice!(
        "Compressing {} vectors ({}-bit, batch={})",
        total, bits, batch
    );

    let mut count: i64 = 0;
    let mut offset: i64 = 0;

    while offset < total {
        // Read batch
        let sql = format!(
            "SELECT id::text, {}::text FROM {} WHERE {} IS NOT NULL \
             ORDER BY id LIMIT {} OFFSET {}",
            source_column, source_table, source_column, batch, offset
        );

        let mut ids: Vec<String> = Vec::new();
        let mut vecs: Vec<Vec<f32>> = Vec::new();

        Spi::connect(|client| {
            let tbl = client.select(&sql, None, None)
                .unwrap_or_else(|e| pgrx::error!("Read: {}", e));
            for row in tbl {
                let id: Option<String> = row.get::<String>(1).ok().flatten();
                let emb_str: Option<String> = row.get::<String>(2).ok().flatten();
                if let (Some(id), Some(emb_s)) = (id, emb_str) {
                    // Parse pgvector text format: [0.1,0.2,...]
                    let cleaned = emb_s.trim_matches(|c| c == '[' || c == ']');
                    let emb: Vec<f32> = cleaned.split(',')
                        .filter_map(|s| s.trim().parse::<f32>().ok())
                        .collect();
                    if !emb.is_empty() {
                        ids.push(id);
                        vecs.push(emb);
                    }
                }
            }
        });

        if vecs.is_empty() {
            break;
        }

        // Batch compress (amortized rotation)
        let compressed = compress::compress_batch(&vecs, bits_u8, 42);

        // Insert compressed vectors
        Spi::connect(|mut client| {
            for (id, tqv) in ids.iter().zip(compressed.iter()) {
                // Serialize tqv to the text output format and insert
                let insert = format!(
                    "INSERT INTO {} (id, tqv) VALUES ('{}', \
                     tq_compress((SELECT {}::float4[] FROM {} WHERE id::text = '{}'), {}))",
                    target_table, id, source_column, source_table, id, bits
                );
                let _ = client.update(&insert, None, None);
            }
        });

        count += ids.len() as i64;
        offset += batch as i64;

        if count % 2000 == 0 || offset >= total {
            pgrx::notice!("Progress: {}/{}", count, total);
        }
    }

    pgrx::notice!("Done: {} vectors compressed", count);
    count
}

// ─── Operator: <=> ───────────────────────────────────────────────

extension_sql!(
    r#"
CREATE OPERATOR <=> (
    LEFTARG = tqvector,
    RIGHTARG = tqvector,
    FUNCTION = tq_cosine_dist,
    COMMUTATOR = <=>
);

COMMENT ON OPERATOR <=> (tqvector, tqvector) IS
    'TurboQuant cosine distance (1 - cosine_similarity)';
"#,
    name = "tqvector_cosine_dist_operator",
);
