pub fn sample_read(buffer: &[f32], position: f32) -> f32 {
    if buffer.is_empty() || position < 0.0 {
        return 0.0;
    }
    let i = position as usize;
    if i + 1 >= buffer.len() {
        if i < buffer.len() {
            return buffer[i];
        }
        return 0.0;
    }
    let frac = position - i as f32;
    buffer[i] + (buffer[i + 1] - buffer[i]) * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    #[test]
    fn integer_position_returns_exact_sample() {
        let buf = [0.0, 0.25, 0.5, 0.75, 1.0];
        for i in 0..buf.len() {
            assert_eq!(sample_read(&buf, i as f32), buf[i], "at {i}");
        }
    }

    #[test]
    fn fractional_position_lerps() {
        let buf = [0.0, 1.0];
        assert!((sample_read(&buf, 0.0) - 0.0).abs() < 1e-6);
        assert!((sample_read(&buf, 0.25) - 0.25).abs() < 1e-6);
        assert!((sample_read(&buf, 0.5) - 0.5).abs() < 1e-6);
        assert!((sample_read(&buf, 0.75) - 0.75).abs() < 1e-6);
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
            assert_eq!(sample_read(&buf, i as f32), buf[i]);
        }

        let s0 = sample_read(&buf, 0.5);
        let s1 = sample_read(&buf, 1.5);
        let s2 = sample_read(&buf, 2.5);
        assert!(s0 < s1 && s1 < s2, "sine should rise across first quarter cycle");
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
            let s1 = sample_read(&buf, n as f32);
            let s2 = sample_read(&buf, (n * 2) as f32);
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
}
