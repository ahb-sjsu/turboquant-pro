//! A small JSON reader for the index `meta` section, so the crate needs no dependency.
//!
//! It accepts what Python's `json.dumps` writes, including the non-standard `NaN`, `Infinity` and `-Infinity`
//! that Python emits by default. Numbers keep their source text, so an integer such as `ids_arange_start` is read
//! exactly rather than through a float.

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// The number's text as written.
    Number(String),
    String(String),
    Array(Vec<Json>),
    /// Members in file order.
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json> {
        let mut p = Parser { s: text.as_bytes(), i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(p.err("trailing characters"));
        }
        Ok(v)
    }

    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(t) => match t.as_str() {
                "NaN" => Some(f64::NAN),
                "Infinity" => Some(f64::INFINITY),
                "-Infinity" => Some(f64::NEG_INFINITY),
                _ => t.parse().ok(),
            },
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Number(t) => t.parse().ok(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Number(t) => t.parse().ok(),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(m) => Some(m),
            _ => None,
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, what: &str) -> Error {
        Error::Json(format!("{what} at byte {}", self.i))
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Json> {
        match self.s.get(self.i).copied() {
            None => Err(self.err("unexpected end")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(b'N') if self.eat("NaN") => Ok(Json::Number("NaN".into())),
            Some(b'I') if self.eat("Infinity") => Ok(Json::Number("Infinity".into())),
            Some(b'-') if self.eat("-Infinity") => Ok(Json::Number("-Infinity".into())),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        }
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.i;
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.i < p.s.len() && p.s[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i > s
        };
        if !digits(self) {
            return Err(self.err("bad number"));
        }
        if self.i < self.s.len() && self.s[self.i] == b'.' {
            self.i += 1;
            if !digits(self) {
                return Err(self.err("bad fraction"));
            }
        }
        if self.i < self.s.len() && matches!(self.s[self.i], b'e' | b'E') {
            self.i += 1;
            if self.i < self.s.len() && matches!(self.s[self.i], b'+' | b'-') {
                self.i += 1;
            }
            if !digits(self) {
                return Err(self.err("bad exponent"));
            }
        }
        Ok(Json::Number(String::from_utf8(self.s[start..self.i].to_vec()).unwrap()))
    }

    fn hex4(&mut self) -> Result<u32> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| self.err("short \\u escape"))?;
        let v = u32::from_str_radix(std::str::from_utf8(h).map_err(|_| self.err("bad \\u escape"))?, 16)
            .map_err(|_| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let c = *self.s.get(self.i).ok_or_else(|| self.err("unterminated string"))?;
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    let e = *self.s.get(self.i).ok_or_else(|| self.err("unterminated escape"))?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !self.eat("\\u") {
                                    return Err(self.err("lone high surrogate"));
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("bad low surrogate"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or_else(|| self.err("bad code point"))?);
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
                _ => {
                    // copy one UTF-8 character
                    let len = match c {
                        0x00..=0x7F => 1,
                        0xC0..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        _ => 4,
                    };
                    let ch = self.s.get(self.i..self.i + len).ok_or_else(|| self.err("bad UTF-8"))?;
                    out.push_str(std::str::from_utf8(ch).map_err(|_| self.err("bad UTF-8"))?);
                    self.i += len;
                }
            }
        }
    }

    fn array(&mut self) -> Result<Json> {
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.eat("]") {
            return Ok(Json::Array(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            if self.eat("]") {
                return Ok(Json::Array(out));
            }
            if !self.eat(",") {
                return Err(self.err("expected , or ]"));
            }
        }
    }

    fn object(&mut self) -> Result<Json> {
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.eat("}") {
            return Ok(Json::Object(out));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.err("expected a key"));
            }
            let k = self.string()?;
            self.ws();
            if !self.eat(":") {
                return Err(self.err("expected :"));
            }
            self.ws();
            let v = self.value()?;
            out.push((k, v));
            self.ws();
            if self.eat("}") {
                return Ok(Json::Object(out));
            }
            if !self.eat(",") {
                return Err(self.err("expected , or }"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Json;

    #[test]
    fn what_python_json_dumps_writes() {
        // json.dumps({"a": [1, -2.5e-3, True, None, "x\u00e9\ud83d\ude00"], "n": NaN, "i": -Infinity, "big": 2**63})
        let t = r#"{"a": [1, -0.0025, true, null, "x\u00e9\ud83d\ude00"], "n": NaN, "i": -Infinity, "big": 9223372036854775808}"#;
        let v = Json::parse(t).unwrap();
        let a = v.get("a").unwrap().as_array().unwrap();
        assert_eq!(a[0].as_i64(), Some(1));
        assert_eq!(a[1].as_f64(), Some(-0.0025));
        assert_eq!(a[2].as_bool(), Some(true));
        assert_eq!(a[3], Json::Null);
        assert_eq!(a[4].as_str(), Some("x\u{e9}\u{1F600}"));
        assert!(v.get("n").unwrap().as_f64().unwrap().is_nan());
        assert_eq!(v.get("i").unwrap().as_f64(), Some(f64::NEG_INFINITY));
        assert_eq!(v.get("big").unwrap().as_u64(), Some(9_223_372_036_854_775_808));
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "\"\\x\"", "1 2", "-", "1.", "\"\\ud800\""] {
            assert!(Json::parse(bad).is_err(), "{bad:?}");
        }
    }
}
