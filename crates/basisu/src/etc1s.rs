//! Basis Universal 073eb6a0: palettes, tables and non-video ETC1S slices.
//! Altered source: checked Rust slices; only glTF's non-video target paths.
//! Copyright 2016–2026 Binomial LLC. See ../licenses/{LICENSE,NOTICE}-basis-universal.
use crate::entropy::{Bits, Huffman};
use crate::{le, range, word, Channels, Ktx2, Model, Result};
mod targets;

#[derive(Clone, Copy, Debug)]
pub enum Target {
    Etc1,
    Etc2Rgba,
    Bc7,
    Rgba8,
}
#[derive(Clone, Copy)]
pub(super) struct Endpoint {
    pub color: [usize; 3],
    pub intensity: usize,
}
// Basis Universal's selector palette retains packed rows, ETC1 bits and flags.
pub(super) struct Selector {
    rows: [u8; 4],
    bytes: [u8; 4],
    lo: u8,
    hi: u8,
    unique: u8,
}
impl Selector {
    fn new(rows: [u8; 4]) -> Self {
        let mut bits = 0u32;
        let mut histogram = 0u8;
        for (y, &row) in rows.iter().enumerate() {
            for x in 0..4 {
                let s = (row >> (x * 2)) & 3;
                histogram |= 1 << s;
                let v = [3, 2, 0, 1][usize::from(s)];
                bits |= (v & 1) << (x * 4 + y);
                bits |= (v >> 1) << (x * 4 + y + 16);
            }
        }
        Self {
            rows,
            bytes: bits.to_be_bytes(),
            lo: histogram.trailing_zeros() as u8,
            hi: (7 - histogram.leading_zeros()) as u8,
            unique: histogram.count_ones() as u8,
        }
    }
    #[inline(always)]
    fn get(&self, i: usize) -> usize {
        usize::from((self.rows[i / 4] >> ((i % 4) * 2)) & 3)
    }
    fn map<T>(&self, mut f: impl FnMut(usize) -> T) -> [T; 16] {
        std::array::from_fn(|i| f(self.get(i)))
    }
}
pub(super) const MODIFIERS: [[i32; 4]; 8] = [
    [-8, -2, 2, 8],
    [-17, -5, 5, 17],
    [-29, -9, 9, 29],
    [-42, -13, 13, 42],
    [-60, -18, 18, 60],
    [-80, -24, 24, 80],
    [-106, -33, 33, 106],
    [-183, -47, 47, 183],
];
impl Endpoint {
    pub fn pixel(self, s: usize) -> [u8; 3] {
        self.color.map(|c| {
            (((c << 3) | (c >> 2)) as i32 + MODIFIERS[self.intensity][s]).clamp(0, 255) as u8
        })
    }
}
pub(crate) struct Tables {
    endpoints: Vec<Endpoint>,
    selectors: Vec<Selector>,
    models: [Huffman; 4],
    history: usize,
}
impl Tables {
    pub(crate) fn new(image: &Ktx2<'_>) -> Result<Self> {
        let global = image.global_data;
        let ne = le(range(global, 0, 2)?);
        let ns = le(range(global, 2, 2)?);
        if ne == 0 || ns == 0 {
            return Err("empty BasisLZ codebooks");
        }
        let mut at = 20 + image.levels.len() * 20;
        let mut payloads = Vec::new();
        for i in 0..4 {
            let n = word(global, 4 + i * 4)?;
            if i < 3 && n == 0 {
                return Err("empty BasisLZ global payload");
            }
            payloads.push(range(global, at, n)?);
            at = at.checked_add(n).ok_or("BasisLZ global size overflow")?;
        }
        let mut b = Bits::new(payloads[0]);
        let colors = [b.table()?, b.table()?, b.table()?];
        let intensity = b.table()?;
        for t in colors.iter().chain([&intensity]) {
            t.require()?;
        }
        let gray = b.get(1)? != 0;
        let mut prev = Endpoint {
            color: [16; 3],
            intensity: 0,
        };
        let mut endpoints = Vec::with_capacity(ne);
        for _ in 0..ne {
            prev.intensity = (prev.intensity + intensity.decode(&mut b)?) & 7;
            for c in 0..if gray { 1 } else { 3 } {
                let model = if prev.color[c] <= 9 {
                    0
                } else if prev.color[c] <= 21 {
                    1
                } else {
                    2
                };
                prev.color[c] = (prev.color[c] + colors[model].decode(&mut b)?) & 31;
            }
            if gray {
                prev.color[1] = prev.color[0];
                prev.color[2] = prev.color[0];
            }
            endpoints.push(prev);
        }
        let mut b = Bits::new(payloads[1]);
        if b.get(1)? != 0 || b.get(1)? != 0 {
            return Err("global selector codebooks unsupported");
        }
        let raw = b.get(1)? != 0;
        let delta = if raw { None } else { Some(b.table()?) };
        if ns > 1 {
            if let Some(t) = &delta {
                t.require()?;
            }
        }
        let mut previous = [0; 4];
        let mut selectors = Vec::with_capacity(ns);
        for i in 0..ns {
            let mut rows = [0; 4];
            for y in 0..4 {
                let byte = if raw || i == 0 {
                    b.get(8)?
                } else {
                    delta.as_ref().unwrap().decode(&mut b)? ^ previous[y]
                };
                if byte > 255 {
                    return Err("invalid selector delta");
                }
                previous[y] = byte;
                rows[y] = byte as u8;
            }
            selectors.push(Selector::new(rows));
        }
        let mut b = Bits::new(payloads[2]);
        let models = [b.table()?, b.table()?, b.table()?, b.table()?];
        for m in &models {
            m.require()?;
        }
        let history = b.get(13)?;
        if history == 0 {
            return Err("empty BasisLZ selector history");
        }
        Ok(Self {
            endpoints,
            selectors,
            models,
            history,
        })
    }
    fn slice(
        &self,
        bytes: &[u8],
        nx: usize,
        ny: usize,
        mut output: impl FnMut(usize, usize, usize),
    ) -> Result<()> {
        let mut b = Bits::new(bytes);
        let mut history = vec![0; self.history];
        let mut rover = self.history / 2;
        let mut rows = [vec![(0, 0); nx], vec![(0, 0); nx]];
        let (mut pred_bits, mut prev_pred, mut repeat, mut prev_endpoint, mut selector_run) =
            (0, 0, 0, 0, 0);
        for y in 0..ny {
            for x in 0..nx {
                let row = y & 1;
                if x & 1 == 0 {
                    if y & 1 == 0 {
                        if repeat != 0 {
                            repeat -= 1;
                            pred_bits = prev_pred;
                        } else {
                            pred_bits = self.models[0].decode(&mut b)?;
                            if pred_bits == 256 {
                                repeat =
                                    b.vlc(4)?.checked_add(2).ok_or("predictor run overflow")?;
                                pred_bits = prev_pred;
                            } else {
                                if pred_bits > 255 {
                                    return Err("invalid endpoint predictor");
                                }
                                prev_pred = pred_bits;
                            }
                        }
                        rows[row ^ 1][x].0 = pred_bits >> 4;
                    } else {
                        pred_bits = rows[row][x].0;
                    }
                }
                let pred = pred_bits & 3;
                pred_bits >>= 2;
                let endpoint = match pred {
                    0 if x != 0 => prev_endpoint,
                    1 if y != 0 => rows[row ^ 1][x].1,
                    2 if x != 0 && y != 0 => rows[row ^ 1][x - 1].1,
                    3 => {
                        let mut v = prev_endpoint + self.models[1].decode(&mut b)?;
                        if v >= self.endpoints.len() {
                            v -= self.endpoints.len();
                        }
                        v
                    }
                    _ => return Err("invalid endpoint prediction"),
                };
                rows[row][x].1 = endpoint;
                prev_endpoint = endpoint;
                let ns = self.selectors.len();
                let sym = if selector_run != 0 {
                    selector_run -= 1;
                    ns
                } else {
                    let s = self.models[2].decode(&mut b)?;
                    if s == ns + self.history {
                        let r = self.models[3].decode(&mut b)?;
                        if r > 63 {
                            return Err("invalid selector run symbol");
                        }
                        selector_run = if r == 63 { b.vlc(7)? } else { r };
                        selector_run =
                            selector_run.checked_add(3).ok_or("selector run overflow")?;
                        if selector_run > nx * ny {
                            return Err("invalid selector run");
                        }
                        selector_run -= 1;
                        ns
                    } else {
                        s
                    }
                };
                let selector = if sym >= ns {
                    let i = sym - ns;
                    let value = *history.get(i).ok_or("invalid selector history index")?;
                    history.swap(i / 2, i);
                    value
                } else {
                    history[rover] = sym;
                    rover += 1;
                    if rover == history.len() {
                        rover = history.len() / 2;
                    }
                    sym
                };
                if endpoint >= self.endpoints.len() || selector >= ns {
                    return Err("invalid BasisLZ codebook index");
                }
                output(y * nx + x, endpoint, selector);
            }
        }
        if repeat != 0 {
            return Err("unfinished endpoint predictor run");
        }
        Ok(())
    }
}
impl Ktx2<'_> {
    /// Transcodes one ETC1S level, tightly packed in block order or RGBA pixel rows.
    /// ETC1 drops alpha; the other targets retain it. Video slices are refused.
    pub fn transcode_etc1s(&self, level: usize, target: Target) -> Result<Vec<u8>> {
        if self.model != Model::Etc1s {
            return Err("not an ETC1S image");
        }
        self.transcode_etc1s_with_tables(level, target, &Tables::new(self)?)
    }
    pub(crate) fn transcode_etc1s_with_tables(
        &self,
        level: usize,
        target: Target,
        tables: &Tables,
    ) -> Result<Vec<u8>> {
        if self.model != Model::Etc1s {
            return Err("not an ETC1S image");
        }
        let data = self.levels.get(level).ok_or("invalid level index")?.bytes;
        let width = (self.width >> level).max(1);
        let height = (self.height >> level).max(1);
        let nx = width.div_ceil(4);
        let ny = height.div_ceil(4);
        // The reference bounds working allocations to 256 MiB.
        let size = match target {
            Target::Rgba8 => width.checked_mul(height).and_then(|n| n.checked_mul(4)),
            Target::Etc1 => nx.checked_mul(ny).and_then(|n| n.checked_mul(8)),
            _ => nx.checked_mul(ny).and_then(|n| n.checked_mul(16)),
        }
        .filter(|&n| n <= 256 * 1024 * 1024)
        .ok_or("ETC1S output too large")?;
        let desc = range(self.global_data, 20 + level * 20, 20)?;
        if word(desc, 0)? != 0 {
            return Err("ETC1S video image flags unsupported");
        }
        let rgb_len = word(desc, 8)?;
        if rgb_len == 0 {
            return Err("missing ETC1S color slice");
        }
        let has_alpha = matches!(self.channels, Channels::Rgba | Channels::RedGreen);
        let mut out = vec![0; size];
        let mut color = Vec::new();
        if matches!(target, Target::Bc7) {
            color.reserve_exact(nx * ny);
        }
        // transcode_slice writes each decoded block directly to its target.
        // Only BC7's subsequent chroma filter needs the color index map.
        tables.slice(range(data, word(desc, 4)?, rgb_len)?, nx, ny, |i, e, s| {
            let endpoint = tables.endpoints[e];
            let selector = &tables.selectors[s];
            match target {
                Target::Etc1 => targets::etc1(&mut out[i * 8..i * 8 + 8], endpoint, selector),
                Target::Etc2Rgba => {
                    targets::etc1(&mut out[i * 16 + 8..i * 16 + 16], endpoint, selector);
                    if !has_alpha {
                        targets::eac(&mut out[i * 16..i * 16 + 8], None);
                    }
                }
                Target::Bc7 => {
                    targets::bc7_color(&mut out[i * 16..i * 16 + 16], endpoint, selector);
                    color.push((e, s));
                }
                Target::Rgba8 => {
                    for y in 0..4 {
                        for x in 0..4 {
                            let px = i % nx * 4 + x;
                            let py = i / nx * 4 + y;
                            if px < width && py < height {
                                let at = (py * width + px) * 4;
                                out[at..at + 3]
                                    .copy_from_slice(&endpoint.pixel(selector.get(y * 4 + x)));
                                out[at + 3] = 255;
                            }
                        }
                    }
                }
            }
        })?;
        if matches!(target, Target::Bc7) {
            targets::chroma_filter(&mut out, nx, ny, &color, &tables.endpoints);
        }
        if has_alpha {
            let n = word(desc, 16)?;
            if n == 0 {
                return Err("missing ETC1S alpha slice");
            }
            tables.slice(range(data, word(desc, 12)?, n)?, nx, ny, |i, e, s| {
                let endpoint = tables.endpoints[e];
                let selector = &tables.selectors[s];
                match target {
                    Target::Etc1 => (),
                    Target::Etc2Rgba => {
                        targets::eac(&mut out[i * 16..i * 16 + 8], Some((endpoint, selector)))
                    }
                    Target::Bc7 => {
                        targets::bc7_alpha(&mut out[i * 16..i * 16 + 16], endpoint, selector)
                    }
                    Target::Rgba8 => {
                        for y in 0..4 {
                            for x in 0..4 {
                                let px = i % nx * 4 + x;
                                let py = i / nx * 4 + y;
                                if px < width && py < height {
                                    out[(py * width + px) * 4 + 3] =
                                        endpoint.pixel(selector.get(y * 4 + x))[1];
                                }
                            }
                        }
                    }
                }
            })?;
        }
        Ok(out)
    }
}
