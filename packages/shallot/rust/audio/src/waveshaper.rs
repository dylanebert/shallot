/// Memoryless waveshaper (soft / hard / fold) with a DC-blocker tail.
///
/// The drive law and soft-clip curve are DaisySP `Effects/overdrive.cpp`'s
/// `SetDrive` + `SoftClip` (MIT): a single `drive in [0,1]` maps to the overdrive's
/// pre-gain (and, for the soft curve, its output-normalizing post-gain), shared by
/// all three modes so `drive` reads the same everywhere. The nonlinearity swaps per
/// mode — soft is the rational-tanh `SoftClip` (cross-checked against dasp-pytorch's
/// true `tanh` in `examples/gen_vectors.rs`), hard is a plain clip, fold is the
/// triangle wrap from DaisySP `Effects/wavefolder.cpp`. Asymmetric shaping introduces
/// DC, so the tail runs DaisySP `Utility/dcblock.cpp`'s one-pole DC blocker.
#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum ShaperMode {
    Soft = 0,
    Hard = 1,
    Fold = 2,
}

impl ShaperMode {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Hard,
            2 => Self::Fold,
            _ => Self::Soft,
        }
    }
}

/// DaisySP `SoftClip`: rational-`tanh` (`SoftLimit`) saturator, hard-limited past ±3.
fn soft_clip(x: f32) -> f32 {
    if x < -3.0 {
        -1.0
    } else if x > 3.0 {
        1.0
    } else {
        x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
    }
}

/// DaisySP `Wavefolder::Process` with unit gain and zero offset (the drive pre-gain
/// is applied by the caller). Triangle wrap into [-1, 1].
fn wavefold(x: f32) -> f32 {
    let ft = ((x + 1.0) * 0.5).floor();
    let sgn = if (ft as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * (x - 2.0 * ft)
}

/// DaisySP `Overdrive::SetDrive`: `drive in [0,1]` → (pre-gain, soft-mode post-gain).
/// The post-gain normalizes the soft curve's output level; hard/fold are already
/// bounded so they use the pre-gain only.
fn drive_gains(drive: f32) -> (f32, f32) {
    let drive = drive.clamp(0.0, 1.0);
    let d = 2.0 * drive;
    let d2 = d * d;
    let pre_a = d * 0.5;
    let pre_b = d2 * d2 * d * 24.0;
    let pre = pre_a + (pre_b - pre_a) * d2;
    let squashed = d * (2.0 - d);
    let post = 1.0 / soft_clip(0.33 + squashed * (pre - 0.33));
    (pre, post)
}

/// Memoryless transfer: drive-scale then the mode's nonlinearity. No DC blocker
/// (that's stateful — see `DcBlock`), so this is the pure curve `gen_vectors` pins.
fn shape(mode: ShaperMode, drive: f32, x: f32) -> f32 {
    let (pre, post) = drive_gains(drive);
    match mode {
        ShaperMode::Soft => soft_clip(pre * x) * post,
        ShaperMode::Hard => (pre * x).clamp(-1.0, 1.0),
        ShaperMode::Fold => wavefold(pre * x),
    }
}

/// DaisySP `DcBlock`: one-pole high-pass, `gain = 1 - 10/sr` (corner ≈ 1.6 Hz @ 48k).
#[derive(Clone, Copy)]
pub struct DcBlock {
    pub x1: f32,
    pub y1: f32,
}

/// `gain = 1 - 10/sample_rate` (DaisySP `DcBlock::Init`).
pub fn dc_gain(sample_rate: f32) -> f32 {
    1.0 - 10.0 / sample_rate
}

impl DcBlock {
    /// One sample of DaisySP `DcBlock::Process`: `out = in - x1 + gain*y1`.
    pub fn tick(&mut self, x: f32, gain: f32) -> f32 {
        let out = x - self.x1 + gain * self.y1;
        self.y1 = out;
        self.x1 = x;
        out
    }
}

/// One sample: shape the input, DC-block the result, crossfade dry/wet at `mix`.
pub fn process(
    dc: &mut DcBlock,
    input: f32,
    mode: ShaperMode,
    drive: f32,
    dc_gain: f32,
    mix: f32,
) -> f32 {
    let shaped = dc.tick(shape(mode, drive, input), dc_gain);
    let mix = mix.clamp(0.0, 1.0);
    input * (1.0 - mix) + shaped * mix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveshaper_golden() {
        // Production's memoryless `shape` bit-checked against the independent
        // gen_vectors transcription (DaisySP overdrive drive law + SoftClip / clip /
        // wavefold) at f32 roundoff over ~10 flops.
        let tol = 1e-5;
        for &(mode, drive, x, want) in crate::golden::WAVESHAPER {
            let got = shape(ShaperMode::from_u32(mode), drive, x);
            assert!(
                (got - want).abs() <= tol,
                "shape(mode={mode}, drive={drive}, x={x}): got {got}, want {want}, Δ{}",
                (got - want).abs()
            );
        }
    }

    #[test]
    fn dc_block_golden() {
        // The DC-blocker recurrence bit-checked against the gen_vectors transcription
        // over the same two-level probe (a step, then a level change to exercise the
        // `in - x1` differencing).
        let tol = 1e-6;
        for &(sr, want) in crate::golden::DC_BLOCK {
            let gain = dc_gain(sr);
            let mut dc = DcBlock { x1: 0.0, y1: 0.0 };
            for (n, w) in want.iter().enumerate() {
                let input = if n < 16 { 1.0 } else { 0.5 };
                let got = dc.tick(input, gain);
                assert!(
                    (got - w).abs() <= tol,
                    "dc_block(sr={sr}) sample {n}: got {got}, want {w}, Δ{}",
                    (got - w).abs()
                );
            }
        }
    }

    #[test]
    fn soft_clip_saturates_and_is_odd() {
        // SoftClip is bounded to [-1,1] and odd-symmetric — the defining properties
        // of a symmetric saturator, independent of the golden bit-check.
        for i in -50..=50 {
            let x = i as f32 * 0.1;
            assert!(soft_clip(x).abs() <= 1.0, "soft_clip({x}) exceeds unity");
            assert!(
                (soft_clip(x) + soft_clip(-x)).abs() < 1e-6,
                "soft_clip should be odd at {x}"
            );
        }
    }

    #[test]
    fn hard_mode_clips_at_unity() {
        // A loud input under hard mode clamps exactly to ±1.
        let hi = shape(ShaperMode::Hard, 0.8, 5.0);
        let lo = shape(ShaperMode::Hard, 0.8, -5.0);
        assert!(
            (hi - 1.0).abs() < 1e-6,
            "hard clip should reach +1, got {hi}"
        );
        assert!(
            (lo + 1.0).abs() < 1e-6,
            "hard clip should reach -1, got {lo}"
        );
    }

    #[test]
    fn fold_mode_stays_bounded_and_folds() {
        // The wavefolder wraps arbitrarily loud input back into [-1,1] rather than
        // clamping — a high-drive ramp must both stay bounded and reverse direction
        // at least once (the fold).
        let mut prev = shape(ShaperMode::Fold, 1.0, -2.0);
        let mut reversals = 0;
        let mut rising = true;
        for i in -19..=20 {
            let x = i as f32 * 0.1;
            let y = shape(ShaperMode::Fold, 1.0, x);
            assert!(
                y.abs() <= 1.0 + 1e-6,
                "fold should stay bounded, got {y} at {x}"
            );
            let now_rising = y >= prev;
            if now_rising != rising {
                reversals += 1;
                rising = now_rising;
            }
            prev = y;
        }
        assert!(
            reversals >= 1,
            "a high-drive ramp should fold at least once"
        );
    }

    #[test]
    fn dc_blocker_removes_offset() {
        // A constant (pure-DC) input decays to zero through the blocker; the AC test
        // is the transfer character above.
        let gain = dc_gain(48000.0);
        let mut dc = DcBlock { x1: 0.0, y1: 0.0 };
        let mut out = 0.0;
        for _ in 0..48000 {
            out = dc.tick(1.0, gain);
        }
        assert!(
            out.abs() < 1e-3,
            "DC blocker should remove a constant offset, got {out}"
        );
    }

    #[test]
    fn mix_zero_bypasses() {
        let gain = dc_gain(48000.0);
        let mut dc = DcBlock { x1: 0.0, y1: 0.0 };
        for i in 0..128 {
            let x = (i as f32 * 0.3).sin();
            let out = process(&mut dc, x, ShaperMode::Soft, 0.8, gain, 0.0);
            assert!(
                (out - x).abs() < 1e-6,
                "mix=0 should bypass, sample {i}: {out} vs {x}"
            );
        }
    }
}
