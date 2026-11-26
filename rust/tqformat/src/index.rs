//! A persisted `TQEIndex` (`turboquant_pro/index.py`): a TQIX container whose `meta` section, JSON, describes
//! every other section as an array (`meta.arrays[name] = {"dtype": ..., "shape": [...]}`, raw C-order bytes).
//!
//! Index format versions 1 to 4 (`docs/FORMATS.md` section 2): v2 adds explicit ids and tombstones, v3 packs the
//! codes at slot granularity (`meta.codes_packed`) and may elide contiguous ids (`meta.ids_arange_start`), v4 is v3
//! with a non-legacy codebook named in `meta.quant.codebook`. Other versions are rejected.

use crate::bits::unpack_rows;
use crate::error::{Error, Result};
use crate::json::Json;
use crate::tqix::{Container, MAGIC_TQIX};

pub const SUPPORTED_VERSIONS: [u16; 4] = [1, 2, 3, 4];

/// An array element type, from a NumPy `dtype.str`. Only little-endian and byte-order-free types are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    F32,
    F64,
}

impl Dtype {
    pub fn parse(s: &str) -> Result<Dtype> {
        Ok(match s {
            "|b1" => Dtype::Bool,
            "|u1" => Dtype::U8,
            "|i1" => Dtype::I8,
            "<u2" => Dtype::U16,
            "<i2" => Dtype::I16,
            "<u4" => Dtype::U32,
            "<i4" => Dtype::I32,
            "<u8" => Dtype::U64,
            "<i8" => Dtype::I64,
            "<f2" => Dtype::F16,
            "<f4" => Dtype::F32,
            "<f8" => Dtype::F64,
            _ => return Err(Error::Meta(format!("unsupported dtype {s:?}"))),
        })
    }

    pub fn itemsize(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
            Dtype::U16 | Dtype::I16 | Dtype::F16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        }
    }
}

/// One array section: its declared type and shape, and its bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Array<'a> {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: &'a [u8],
}

macro_rules! typed {
    ($fn:ident, $t:ty, $dt:path) => {
        pub fn $fn(&self) -> Result<Vec<$t>> {
            if self.dtype != $dt {
                return Err(Error::Meta(format!("{} is {:?}, not {}", self.name, self.dtype, stringify!($t))));
            }
            Ok(self
                .bytes
                .chunks_exact(std::mem::size_of::<$t>())
                .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
    };
}

impl Array<'_> {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    typed!(to_f32, f32, Dtype::F32);
    typed!(to_f64, f64, Dtype::F64);
    typed!(to_i64, i64, Dtype::I64);
    typed!(to_u64, u64, Dtype::U64);
    typed!(to_u32, u32, Dtype::U32);
    typed!(to_i32, i32, Dtype::I32);

    pub fn to_u8(&self) -> Result<Vec<u8>> {
        match self.dtype {
            Dtype::U8 | Dtype::Bool => Ok(self.bytes.to_vec()),
            d => Err(Error::Meta(format!("{} is {d:?}, not u8", self.name))),
        }
    }
}

/// An opened index: the verified container and its parsed `meta`.
#[derive(Debug, Clone)]
pub struct Index {
    pub container: Container,
    pub meta: Json,
}

impl Index {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Index> {
        Index::parse(std::fs::read(path)?)
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Index> {
        let container = Container::parse(bytes, MAGIC_TQIX)?;
        let v = container.version();
        if !SUPPORTED_VERSIONS.contains(&v) {
            return Err(Error::UnsupportedVersion { format: "TQIX index", version: v as u32 });
        }
        let raw = container.section("meta").ok_or_else(|| Error::MissingSection("meta".into()))?;
        let text = std::str::from_utf8(raw).map_err(|e| Error::Json(e.to_string()))?;
        let meta = Json::parse(text)?;
        if meta.get("arrays").and_then(Json::as_object).is_none() {
            return Err(Error::Meta("no arrays table".into()));
        }
        Ok(Index { container, meta })
    }

    /// The container's version (the index format version).
    pub fn format_version(&self) -> u16 {
        self.container.version()
    }

    /// Names of the array sections, as `meta.arrays` lists them.
    pub fn array_names(&self) -> Vec<String> {
        self.meta.get("arrays").and_then(Json::as_object).unwrap().iter().map(|(k, _)| k.clone()).collect()
    }

    /// The array section `name`, checked against its declared dtype and shape.
    pub fn array(&self, name: &str) -> Result<Array<'_>> {
        let spec = self
            .meta
            .get("arrays")
            .and_then(|a| a.get(name))
            .ok_or_else(|| Error::Meta(format!("no array spec for {name:?}")))?;
        let dtype = Dtype::parse(
            spec.get("dtype").and_then(Json::as_str).ok_or_else(|| Error::Meta(format!("{name}: no dtype")))?,
        )?;
        let shape = spec
            .get("shape")
            .and_then(Json::as_array)
            .ok_or_else(|| Error::Meta(format!("{name}: no shape")))?
            .iter()
            .map(|d| d.as_u64().map(|v| v as usize).ok_or_else(|| Error::Meta(format!("{name}: bad shape"))))
            .collect::<Result<Vec<usize>>>()?;
        let bytes = self.container.section(name).ok_or_else(|| Error::MissingSection(name.into()))?;
        let want = shape.iter().product::<usize>() * dtype.itemsize();
        if bytes.len() != want {
            return Err(Error::Meta(format!("{name}: {} bytes, but dtype and shape say {want}", bytes.len())));
        }
        Ok(Array { name: name.into(), dtype, shape, bytes })
    }

    /// The quantizer codes, one row per stored vector, unpacked to one `u8` per code: `(codes, n_rows, dim)`.
    pub fn codes(&self) -> Result<(Vec<u8>, usize, usize)> {
        let a = self.array("codes")?;
        match self.meta.get("codes_packed") {
            None => {
                if a.shape.len() != 2 {
                    return Err(Error::Meta("codes is not two-dimensional".into()));
                }
                Ok((a.to_u8()?, a.shape[0], a.shape[1]))
            }
            Some(cp) => {
                let dim = cp.get("dim").and_then(Json::as_u64).ok_or_else(|| Error::Meta("codes_packed.dim".into()))?
                    as usize;
                let slot = cp
                    .get("slot_bits")
                    .and_then(Json::as_u64)
                    .ok_or_else(|| Error::Meta("codes_packed.slot_bits".into()))? as u8;
                if a.shape.len() != 2 {
                    return Err(Error::Meta("packed codes are not two-dimensional".into()));
                }
                let (rows, cols) = (a.shape[0], a.shape[1]);
                Ok((unpack_rows(a.bytes, rows, cols, dim, slot)?, rows, dim))
            }
        }
    }

    fn n_rows(&self) -> Result<usize> {
        Ok(self.meta.get("n_rows").and_then(Json::as_u64).ok_or_else(|| Error::Meta("n_rows".into()))? as usize)
    }

    /// The row ids, as `TQEIndex.open` reconstructs them: from `ids_arange_start` when a v3 index elided them, the
    /// `ids` section for v2 and later, positional for v1.
    pub fn ids(&self) -> Result<Vec<i64>> {
        if let Some(start) = self.meta.get("ids_arange_start") {
            let start = start.as_i64().ok_or_else(|| Error::Meta("ids_arange_start".into()))?;
            return Ok((start..start + self.n_rows()? as i64).collect());
        }
        if self.format_version() >= 2 {
            return self.array("ids")?.to_i64();
        }
        Ok((0..self.n_rows()? as i64).collect())
    }

    /// The tombstone flags (1 = deleted), reconstructed the same way as [`Index::ids`].
    pub fn tombstones(&self) -> Result<Vec<u8>> {
        if self.meta.get("ids_arange_start").is_none() && self.format_version() >= 2 {
            return self.array("tombstones")?.to_u8();
        }
        Ok(vec![0; self.n_rows()?])
    }
}
