use crate::interp::hermite4;

/// One decoded sample asset: up to two planar per-channel buffers plus the
/// channel count. `count == 1` is mono (a channel-1 read falls back to channel
/// 0), `count == 2` is stereo, `count == 0` is unallocated (an empty buffer,
/// read as silence). The host allocates one channel at a time (`sample_alloc`).
#[derive(Default)]
pub struct SampleBuffer {
    pub channels: [Vec<f32>; 2],
    pub count: u8,
}

/// The read `position` is f64, not f32, so the integer index and fraction stay
/// exact in long buffers — past ~16.78M samples (2²⁴) an f32 position can't
/// resolve adjacent indices and the Hermite read collapses to nearest-neighbor.
/// The buffer itself stays f32.
pub fn sample_read(buffer: &[f32], position: f64) -> f32 {
    let len = buffer.len();
    if len == 0 || position < 0.0 {
        return 0.0;
    }
    let i = position as usize;
    if i + 1 >= len {
        // at or past the last sample: hold the final value, silence beyond it
        return if i < len { buffer[i] } else { 0.0 };
    }
    let frac = (position - i as f64) as f32;
    // edge taps clamp at the buffer ends (a one-shot has no samples outside it)
    let ym1 = if i >= 1 { buffer[i - 1] } else { buffer[0] };
    let y2 = if i + 2 < len {
        buffer[i + 2]
    } else {
        buffer[i + 1]
    };
    hermite4(ym1, buffer[i], buffer[i + 1], y2, frac)
}

/// Loop-aware read: the Hermite taps wrap `mod len` instead of clamping at the
/// buffer ends, so a seam-matched loop is C¹-continuous across the wrap (no flat
/// sample, no derivative break — the tick `sample_read`'s edge clamp leaves). The
/// caller (`sample_node`) already wraps the read position into `[0, len)`;
/// `rem_euclid` guards a defensive out-of-range or negative position. Same f64
/// position contract as `sample_read`.
pub fn sample_read_loop(buffer: &[f32], position: f64) -> f32 {
    let len = buffer.len();
    if len == 0 {
        return 0.0;
    }
    let pos = position.rem_euclid(len as f64);
    let i = (pos as usize).min(len - 1);
    let frac = (pos - i as f64) as f32;
    let ym1 = buffer[(i + len - 1) % len];
    let y1 = buffer[(i + 1) % len];
    let y2 = buffer[(i + 2) % len];
    hermite4(ym1, buffer[i], y1, y2, frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    #[test]
    fn integer_position_returns_exact_sample() {
        let buf = [0.0, 0.25, 0.5, 0.75, 1.0];
        for i in 0..buf.len() {
            assert_eq!(sample_read(&buf, i as f64), buf[i], "at {i}");
        }
    }

    #[test]
    fn linear_ramp_reads_exactly() {
        // Hermite reproduces a line exactly, so reading a linear ramp at any
        // interior fractional position returns the exact ramp value (all four
        // taps lie on the line). Bound to positions where i+2 < len so no edge
        // clamp enters. Tolerance is f32 roundoff.
        let buf: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        for k in 0..50 {
            let pos = 1.0 + k as f32 / 10.0; // [1.0, 5.9] — all four taps real
            assert!(
                (sample_read(&buf, pos as f64) - pos * 0.5).abs() < 1e-5,
                "pos {pos}"
            );
        }
    }

    #[test]
    fn fractional_position_interpolates_monotonically() {
        let buf = [0.0, 1.0];
        assert!((sample_read(&buf, 0.0) - 0.0).abs() < 1e-6);
        let a = sample_read(&buf, 0.25);
        let b = sample_read(&buf, 0.5);
        let c = sample_read(&buf, 0.75);
        assert!(
            a > 0.0 && a < b && b < c && c < 1.0,
            "should rise monotonically through (0,1): {a} {b} {c}"
        );
    }

    #[test]
    fn negative_position_returns_zero() {
        let buf = [1.0, 1.0, 1.0];
        assert_eq!(sample_read(&buf, -0.1), 0.0);
        assert_eq!(sample_read(&buf, -1.0), 0.0);
    }

    #[test]
    fn past_end_returns_zero() {
        let buf = [1.0, 2.0, 3.0];
        assert_eq!(sample_read(&buf, 2.0), 3.0);
        assert_eq!(sample_read(&buf, 3.0), 0.0);
        assert_eq!(sample_read(&buf, 100.0), 0.0);
    }

    #[test]
    fn empty_buffer_returns_zero() {
        let buf: [f32; 0] = [];
        assert_eq!(sample_read(&buf, 0.0), 0.0);
        assert_eq!(sample_read(&buf, 5.5), 0.0);
    }

    #[test]
    fn unit_rate_reproduces_sine() {
        let sr = 48000.0_f32;
        let freq = 480.0_f32;
        let len = 4800;
        let buf: Vec<f32> = (0..len)
            .map(|i| (TAU * freq * i as f32 / sr).sin())
            .collect();

        for i in 0..len {
            assert_eq!(sample_read(&buf, i as f64), buf[i]);
        }

        let s0 = sample_read(&buf, 0.5);
        let s1 = sample_read(&buf, 1.5);
        let s2 = sample_read(&buf, 2.5);
        assert!(
            s0 < s1 && s1 < s2,
            "sine should rise across first quarter cycle"
        );
    }

    #[test]
    fn double_rate_advances_twice_as_fast() {
        let sr = 48000.0_f32;
        let freq = 240.0_f32;
        let len = 4800;
        let buf: Vec<f32> = (0..len)
            .map(|i| (TAU * freq * i as f32 / sr).sin())
            .collect();

        let blocks = 1000;
        let mut crossings_1x = 0;
        let mut crossings_2x = 0;
        let mut prev_1x = sample_read(&buf, 0.0);
        let mut prev_2x = sample_read(&buf, 0.0);
        for n in 1..blocks {
            let s1 = sample_read(&buf, n as f64);
            let s2 = sample_read(&buf, (n * 2) as f64);
            if (prev_1x < 0.0) != (s1 < 0.0) {
                crossings_1x += 1;
            }
            if (prev_2x < 0.0) != (s2 < 0.0) {
                crossings_2x += 1;
            }
            prev_1x = s1;
            prev_2x = s2;
        }
        assert!(
            crossings_2x as f32 >= 1.8 * crossings_1x as f32,
            "crossings 1x={crossings_1x} 2x={crossings_2x}",
        );
    }

    #[test]
    fn long_buffer_position_tracks_analytic() {
        // Past ~4.2M samples an f32 read position can no longer represent the
        // fraction between adjacent samples (spacing ≥ 0.5), so the Hermite
        // read snaps toward nearest-neighbor and drifts from the analytic
        // value; an f64 position keeps the fraction exact. Tolerance is the
        // Hermite truncation + f32 storage error for this oscillation (~1e-6),
        // well below the ~omega/2 drift an f32 position introduces.
        let omega = 0.05_f64;
        let n = 6_000_000usize;
        let buf: Vec<f32> = (0..n).map(|i| (omega * i as f64).sin() as f32).collect();

        let base = 5_000_000.0_f64;
        let mut max_err = 0.0_f64;
        for k in 0..32 {
            let p = base + k as f64 * 0.5 + 0.25;
            let got = sample_read(&buf, p) as f64;
            let want = (omega * p).sin();
            max_err = max_err.max((got - want).abs());
        }
        assert!(
            max_err < 1e-3,
            "long-buffer read should track the analytic sine, max error {max_err}"
        );
    }

    #[test]
    fn loop_read_is_c1_continuous_across_wrap() {
        // A single-cycle sine seam-matches exactly: buffer[i] = sin(TAU*i/N) is
        // C∞-periodic with period N, so value AND derivative are continuous across
        // the wrap by construction. The loop-aware read must reproduce the analytic
        // sine through the seam; matching it proves both failure modes of the old
        // edge clamp are gone — no flat-held sample (value matches) and no
        // derivative break (the curve is followed, not clamped). Tolerance is the
        // sample.rs oscillation-read convention: omega = TAU/N ≈ 0.049 rad/sample
        // matches long_buffer_position_tracks_analytic's 0.05 regime, whose Hermite
        // truncation + f32 error stays under 1e-3.
        let n = 128usize;
        let buf: Vec<f32> = (0..n).map(|i| (TAU * i as f32 / n as f32).sin()).collect();
        let mut pos = n as f64 - 2.0;
        while pos < n as f64 + 2.0 {
            let got = sample_read_loop(&buf, pos) as f64;
            let want = (TAU as f64 * pos / n as f64).sin();
            assert!(
                (got - want).abs() < 1e-3,
                "pos {pos}: got {got}, want {want}"
            );
            pos += 0.1;
        }
    }

    #[test]
    fn loop_read_wraps_taps() {
        // At i=0 the ym1 tap must come from buffer[len-1] (wrap), not buffer[0]
        // (clamp). Ends chosen far apart so the choice is unambiguous in the read.
        let buf = [0.0, 1.0, 2.0, 9.0]; // buffer[0]=0, buffer[len-1]=9
        let frac = 0.25f32;
        let want = crate::interp::hermite4(buf[3], buf[0], buf[1], buf[2], frac);
        let got = sample_read_loop(&buf, 0.25);
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        let clamp = crate::interp::hermite4(buf[0], buf[0], buf[1], buf[2], frac);
        assert!(
            (got - clamp).abs() > 1e-3,
            "wrapped ym1 must differ from a clamped ym1"
        );
    }

    #[test]
    fn loop_read_differs_from_clamp_at_seam() {
        // Guard that the loop path actually changed the seam, so the C¹ test isn't
        // green for the wrong reason: at the last fractional sample before the wrap
        // the clamping sample_read holds buffer[n-1] while sample_read_loop
        // interpolates across to buffer[0..]; on a sine they must disagree.
        let n = 128usize;
        let buf: Vec<f32> = (0..n).map(|i| (TAU * i as f32 / n as f32).sin()).collect();
        let pos = n as f64 - 0.5;
        let clamp = sample_read(&buf, pos);
        let looped = sample_read_loop(&buf, pos);
        assert!(
            (clamp - looped).abs() > 1e-3,
            "seam read should differ: clamp {clamp}, loop {looped}"
        );
    }
}
