-- tqvector extension v0.1.0
-- TurboQuant compressed vector type for PostgreSQL (Rust/pgrx)

CREATE TYPE tqvector;

CREATE FUNCTION tqvector_in(cstring) RETURNS tqvector
AS '$libdir/tqvector', 'tqvector_in_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tqvector_out(tqvector) RETURNS cstring
AS '$libdir/tqvector', 'tqvector_out_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE tqvector (
    INPUT = tqvector_in,
    OUTPUT = tqvector_out,
    STORAGE = extended
);

CREATE FUNCTION tq_compress(float4[], int DEFAULT 3) RETURNS tqvector
AS '$libdir/tqvector', 'tq_compress_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tq_decompress(tqvector) RETURNS float4[]
AS '$libdir/tqvector', 'tq_decompress_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tq_cosine_sim(tqvector, tqvector) RETURNS float4
AS '$libdir/tqvector', 'tq_cosine_sim_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tq_cosine_dist(tqvector, tqvector) RETURNS float4
AS '$libdir/tqvector', 'tq_cosine_dist_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tq_dim(tqvector) RETURNS int
AS '$libdir/tqvector', 'tq_dim_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tq_bits(tqvector) RETURNS int
AS '$libdir/tqvector', 'tq_bits_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tq_ratio(tqvector) RETURNS float4
AS '$libdir/tqvector', 'tq_ratio_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tq_norm(tqvector) RETURNS float4
AS '$libdir/tqvector', 'tq_norm_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tq_size_bytes(tqvector) RETURNS int
AS '$libdir/tqvector', 'tq_size_bytes_wrapper' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tq_bulk_compress(text, text, text, int DEFAULT 3, int DEFAULT 500) RETURNS int8
AS '$libdir/tqvector', 'tq_bulk_compress_wrapper' LANGUAGE C;

COMMENT ON FUNCTION tq_bulk_compress(text, text, text, int, int) IS
    'Bulk compress: source_table, source_col, target_table, bits, batch_size';

CREATE OPERATOR <=> (
    LEFTARG = tqvector, RIGHTARG = tqvector,
    FUNCTION = tq_cosine_dist, COMMUTATOR = <=>
);
