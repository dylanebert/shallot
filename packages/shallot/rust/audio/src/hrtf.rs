use core::f32::consts::PI;

pub const NUM_SPEAKERS: usize = 8;
pub const HRTF_TAPS: usize = 128;

pub struct Speaker {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub left: [f32; HRTF_TAPS],
    pub right: [f32; HRTF_TAPS],
}

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

fn hann(i: usize, n: usize) -> f32 {
    0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos())
}

fn build_delay_filter(delay_samples: f32, gain: f32, buf: &mut [f32; HRTF_TAPS]) {
    let center = (HRTF_TAPS / 2) as f32;
    let d = center + delay_samples;
    for i in 0..HRTF_TAPS {
        buf[i] = gain * sinc(i as f32 - d) * hann(i, HRTF_TAPS);
    }
}

fn brown_duda_alpha(theta: f32) -> f32 {
    const ALPHA_MIN: f32 = 0.1;
    const THETA_MIN: f32 = 5.0 * PI / 6.0;
    if theta < THETA_MIN {
        (1.0 + ALPHA_MIN) / 2.0 + (1.0 - ALPHA_MIN) / 2.0 * (theta * PI / THETA_MIN).cos()
    } else {
        ALPHA_MIN
    }
}

fn apply_head_shadow(buf: &mut [f32; HRTF_TAPS], alpha: f32, tau: f32, sample_rate: f32) {
    let c = 2.0 * sample_rate;
    let denom = tau * c + 1.0;
    let b0 = (alpha * tau * c + 1.0) / denom;
    let b1 = (1.0 - alpha * tau * c) / denom;
    let a1 = (1.0 - tau * c) / denom;

    let mut prev_in: f32 = 0.0;
    let mut prev_out: f32 = 0.0;
    for i in 0..HRTF_TAPS {
        let x = buf[i];
        let y = b0 * x + b1 * prev_in - a1 * prev_out;
        buf[i] = y;
        prev_in = x;
        prev_out = y;
    }
}

fn compute_speaker(az_deg: f32, el_deg: f32, sample_rate: f32) -> Speaker {
    let az = az_deg * PI / 180.0;
    let el = el_deg * PI / 180.0;

    let x = az.sin() * el.cos();
    let y = el.sin();
    let z = az.cos() * el.cos();

    let head_radius: f32 = 0.0875;
    let speed_of_sound: f32 = 343.0;
    let tau = 2.0 * head_radius / speed_of_sound;

    let angle_left = az - PI * 0.5;
    let angle_right = az + PI * 0.5;

    let itd_left = head_radius / speed_of_sound * (angle_left.cos().max(-1.0).min(1.0) + 1.0);
    let itd_right = head_radius / speed_of_sound * (angle_right.cos().max(-1.0).min(1.0) + 1.0);

    let delay_left = itd_left * sample_rate;
    let delay_right = itd_right * sample_rate;

    let theta_left = (-x).acos();
    let theta_right = x.acos();
    let alpha_left = brown_duda_alpha(theta_left);
    let alpha_right = brown_duda_alpha(theta_right);

    let mut left = [0.0f32; HRTF_TAPS];
    let mut right = [0.0f32; HRTF_TAPS];
    build_delay_filter(delay_left, 1.0, &mut left);
    build_delay_filter(delay_right, 1.0, &mut right);
    apply_head_shadow(&mut left, alpha_left, tau, sample_rate);
    apply_head_shadow(&mut right, alpha_right, tau, sample_rate);

    Speaker {
        x,
        y,
        z,
        left,
        right,
    }
}

#[cfg(test)]
pub(crate) fn test_brown_duda_alpha(theta: f32) -> f32 {
    brown_duda_alpha(theta)
}

#[cfg(test)]
pub(crate) fn test_build_delay_filter(delay_samples: f32, gain: f32, buf: &mut [f32; HRTF_TAPS]) {
    build_delay_filter(delay_samples, gain, buf);
}

#[cfg(test)]
pub(crate) fn test_compute_speaker(az_deg: f32, el_deg: f32, sample_rate: f32) -> Speaker {
    compute_speaker(az_deg, el_deg, sample_rate)
}

pub fn init_speakers(sample_rate: f32) -> [Speaker; NUM_SPEAKERS] {
    let dirs: [(f32, f32); NUM_SPEAKERS] = [
        (45.0, 35.26),
        (135.0, 35.26),
        (225.0, 35.26),
        (315.0, 35.26),
        (45.0, -35.26),
        (135.0, -35.26),
        (225.0, -35.26),
        (315.0, -35.26),
    ];

    let mut speakers: [Speaker; NUM_SPEAKERS] = unsafe { core::mem::zeroed() };
    for (i, &(az, el)) in dirs.iter().enumerate() {
        speakers[i] = compute_speaker(az, el, sample_rate);
    }
    speakers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_positions_unit_sphere() {
        let speakers = init_speakers(48000.0);
        for (i, s) in speakers.iter().enumerate() {
            let r = (s.x * s.x + s.y * s.y + s.z * s.z).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-4,
                "speaker {i}: radius {r}, expected ~1.0"
            );
        }
    }

    #[test]
    fn speaker_positions_symmetric() {
        let speakers = init_speakers(48000.0);
        for i in 0..4 {
            let upper = &speakers[i];
            let lower = &speakers[i + 4];
            assert!(
                (upper.x - lower.x).abs() < 1e-4,
                "pair {i}: x mismatch {} vs {}",
                upper.x,
                lower.x
            );
            assert!(
                (upper.y + lower.y).abs() < 1e-4,
                "pair {i}: y not mirrored {} vs {}",
                upper.y,
                lower.y
            );
            assert!(
                (upper.z - lower.z).abs() < 1e-4,
                "pair {i}: z mismatch {} vs {}",
                upper.z,
                lower.z
            );
        }
    }

    #[test]
    fn itd_zero_for_frontal_source() {
        let speaker = test_compute_speaker(0.0, 0.0, 48000.0);
        let left_peak = speaker
            .left
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let right_peak = speaker
            .right
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let diff = (left_peak as i32 - right_peak as i32).abs();
        assert!(
            diff <= 1,
            "frontal source: left/right peak offset should be ≤1, got {diff}"
        );
    }

    #[test]
    fn itd_maximum_for_lateral_source() {
        let speaker = test_compute_speaker(90.0, 0.0, 48000.0);
        let left_peak = speaker
            .left
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let right_peak = speaker
            .right
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let diff = (left_peak as i32 - right_peak as i32).abs();
        let expected = (0.0875f32 / 343.0 * 2.0 * 48000.0).round() as i32;
        assert!(
            diff >= 3,
            "lateral source: expected meaningful ITD offset (~{expected}), got {diff}"
        );
    }

    #[test]
    fn ild_contralateral_attenuation() {
        let speaker = test_compute_speaker(90.0, 0.0, 48000.0);
        let left_energy: f32 = speaker.left.iter().map(|s| s * s).sum();
        let right_energy: f32 = speaker.right.iter().map(|s| s * s).sum();
        let (ipsi, contra) = if left_energy > right_energy {
            (left_energy, right_energy)
        } else {
            (right_energy, left_energy)
        };
        assert!(
            ipsi > contra * 1.5,
            "lateral source: ipsilateral energy {ipsi} should be >1.5x contralateral {contra}"
        );
    }

    #[test]
    fn filter_energy_bounded() {
        let speakers = init_speakers(48000.0);
        for (i, s) in speakers.iter().enumerate() {
            let left_energy: f32 = s.left.iter().map(|v| v * v).sum();
            let right_energy: f32 = s.right.iter().map(|v| v * v).sum();
            assert!(
                left_energy <= 1.0 + 0.05,
                "speaker {i} left energy {left_energy} exceeds 1.0"
            );
            assert!(
                right_energy <= 1.0 + 0.05,
                "speaker {i} right energy {right_energy} exceeds 1.0"
            );
        }
    }

    #[test]
    fn brown_duda_alpha_range() {
        for deg in 0..=180 {
            let theta = deg as f32 * PI / 180.0;
            let alpha = test_brown_duda_alpha(theta);
            assert!(
                alpha >= 0.1 - 1e-4 && alpha <= 1.0 + 1e-4,
                "alpha({deg}°) = {alpha}, out of [0.1, 1.0]"
            );
        }
        let frontal = test_brown_duda_alpha(0.0);
        assert!(
            (frontal - 1.0).abs() < 0.01,
            "alpha(0) should be ~1.0, got {frontal}"
        );
    }

    #[test]
    fn sinc_filter_normalized() {
        let mut buf = [0.0f32; HRTF_TAPS];
        test_build_delay_filter(0.0, 1.0, &mut buf);
        let energy: f32 = buf.iter().map(|s| s * s).sum();
        assert!(
            (energy - 1.0).abs() < 0.1,
            "windowed sinc at delay=0, gain=1: energy={energy}, expected ~1.0"
        );
    }
}
