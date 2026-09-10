#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnvStage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

#[derive(Clone, Copy)]
pub struct Envelope {
    pub stage: EnvStage,
    pub level: f32,
    pub time: f32,
    pub attack_start_level: f32,
    pub release_start_level: f32,
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,
    pub attack_curve: f32,
    pub decay_curve: f32,
    pub release_curve: f32,
}

impl Default for Envelope {
    fn default() -> Self {
        Self {
            stage: EnvStage::Idle,
            level: 0.0,
            time: 0.0,
            attack_start_level: 0.0,
            release_start_level: 0.0,
            attack: 0.01,
            decay: 0.1,
            sustain: 0.7,
            release: 0.3,
            attack_curve: 0.0,
            decay_curve: -0.3,
            release_curve: -0.3,
        }
    }
}

impl Envelope {
    pub fn tick(&mut self, dt: f32) -> f32 {
        match self.stage {
            EnvStage::Idle => {
                self.level = 0.0;
            }
            EnvStage::Attack => {
                self.time += dt;
                let t = if self.attack > 0.0 {
                    (self.time / self.attack).min(1.0)
                } else {
                    1.0
                };
                self.level = self.attack_start_level
                    + (1.0 - self.attack_start_level) * curve(t, self.attack_curve);
                if t >= 1.0 {
                    self.level = 1.0;
                    self.stage = EnvStage::Decay;
                    self.time = 0.0;
                }
            }
            EnvStage::Decay => {
                self.time += dt;
                let t = if self.decay > 0.0 {
                    (self.time / self.decay).min(1.0)
                } else {
                    1.0
                };
                self.level = 1.0 + (self.sustain - 1.0) * curve(t, self.decay_curve);
                if t >= 1.0 {
                    self.level = self.sustain;
                    if self.sustain <= 1e-4 {
                        self.release_start_level = self.level;
                        self.stage = EnvStage::Release;
                        self.time = 0.0;
                    } else {
                        self.stage = EnvStage::Sustain;
                    }
                }
            }
            EnvStage::Sustain => {
                self.level = self.sustain;
            }
            EnvStage::Release => {
                self.time += dt;
                let t = if self.release > 0.0 {
                    (self.time / self.release).min(1.0)
                } else {
                    1.0
                };
                self.level = self.release_start_level * (1.0 - curve(t, self.release_curve));
                if t >= 1.0 {
                    self.level = 0.0;
                    self.stage = EnvStage::Idle;
                }
            }
        }
        self.level
    }
}

/// Envelope segment shaping. `c` in [-1, 1] bends a normalized 0→1 segment:
/// `c = 0` is linear, `c > 0` convex (slow then fast), `c < 0` concave. The
/// exponential form `(e^(p·t) - 1)/(e^p - 1)` (p = 6c) maps [0,1] onto [0,1]
/// for any `c`, so endpoints stay exact and the segment is monotonic. This
/// curve shape is the deliberate, settled choice — a single tunable per stage,
/// not a placeholder for per-segment spline tables.
pub fn curve(t: f32, c: f32) -> f32 {
    let p = c * 6.0;
    if p.abs() < 0.01 {
        return t;
    }
    ((p * t).exp() - 1.0) / (p.exp() - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_envelope(env: &mut Envelope, dt: f32, samples: usize) -> f32 {
        let mut last = 0.0;
        for _ in 0..samples {
            last = env.tick(dt);
        }
        last
    }

    #[test]
    fn lifecycle_attack_decay_sustain_release_idle() {
        let mut env = Envelope {
            attack: 0.01,
            decay: 0.01,
            sustain: 0.5,
            release: 0.01,
            ..Envelope::default()
        };
        env.stage = EnvStage::Attack;
        env.time = 0.0;

        run_envelope(&mut env, 0.001, 11);
        assert_eq!(env.stage, EnvStage::Decay);

        run_envelope(&mut env, 0.001, 11);
        assert_eq!(env.stage, EnvStage::Sustain);
        assert!((env.level - 0.5).abs() < 0.01);

        env.release_start_level = env.level;
        env.stage = EnvStage::Release;
        env.time = 0.0;

        run_envelope(&mut env, 0.001, 11);
        assert_eq!(env.stage, EnvStage::Idle);
        assert!(env.level.abs() < 0.01);
    }

    #[test]
    fn zero_sustain_skips_to_release() {
        let mut env = Envelope {
            attack: 0.01,
            decay: 0.01,
            sustain: 0.0,
            release: 0.01,
            ..Envelope::default()
        };
        env.stage = EnvStage::Attack;
        env.time = 0.0;

        run_envelope(&mut env, 0.001, 11);
        assert_eq!(env.stage, EnvStage::Decay);

        run_envelope(&mut env, 0.001, 11);
        assert_eq!(env.stage, EnvStage::Release);

        run_envelope(&mut env, 0.001, 11);
        assert_eq!(env.stage, EnvStage::Idle);
    }

    #[test]
    fn curve_endpoints() {
        assert!((curve(0.0, 0.5) - 0.0).abs() < 1e-6);
        assert!((curve(1.0, 0.5) - 1.0).abs() < 1e-6);
        assert!((curve(0.0, -0.5) - 0.0).abs() < 1e-6);
        assert!((curve(1.0, -0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn curve_monotonically_increasing() {
        for &c in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let mut prev = curve(0.0, c);
            for i in 1..=100 {
                let t = i as f32 / 100.0;
                let val = curve(t, c);
                assert!(
                    val >= prev - 1e-6,
                    "curve(c={c}) not monotonic at t={t}: {val} < {prev}"
                );
                prev = val;
            }
        }
    }

    #[test]
    fn curve_shape_positive_c_convex() {
        let val = curve(0.5, 1.0);
        assert!(
            val < 0.15,
            "c=1.0 at t=0.5 should be <0.15 (slow start), got {val}"
        );
        assert!(val > 0.01, "c=1.0 at t=0.5 should be >0.01, got {val}");
    }

    #[test]
    fn curve_shape_negative_c_concave() {
        let val = curve(0.5, -1.0);
        assert!(
            val > 0.85,
            "c=-1.0 at t=0.5 should be >0.85 (fast start), got {val}"
        );
        assert!(val < 0.99, "c=-1.0 at t=0.5 should be <0.99, got {val}");
    }

    #[test]
    fn attack_no_discontinuity() {
        let mut env = Envelope {
            attack: 0.05,
            decay: 0.1,
            sustain: 0.7,
            release: 0.3,
            ..Envelope::default()
        };
        env.stage = EnvStage::Attack;
        env.time = 0.0;
        env.level = 0.0;

        let dt = 1.0 / 48000.0;
        let expected_step = 1.0 / (0.05 * 48000.0);
        let mut prev = env.tick(dt);
        let mut max_diff = 0.0f32;
        for _ in 1..2400 {
            if env.stage != EnvStage::Attack {
                break;
            }
            let cur = env.tick(dt);
            let diff = (cur - prev).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            prev = cur;
        }
        assert!(
            max_diff < 2.0 * expected_step + 1e-6,
            "attack max step {max_diff} > 2x expected {expected_step}"
        );
    }

    #[test]
    fn decay_sustain_continuity() {
        let mut env = Envelope {
            attack: 0.001,
            decay: 0.05,
            sustain: 0.6,
            release: 0.3,
            ..Envelope::default()
        };
        env.stage = EnvStage::Attack;
        env.time = 0.0;
        env.level = 0.0;

        let dt = 1.0 / 48000.0;
        for _ in 0..50000 {
            env.tick(dt);
            if env.stage == EnvStage::Sustain {
                assert!(
                    (env.level - 0.6).abs() < 0.01,
                    "level at sustain entry should be ~0.6, got {}",
                    env.level
                );
                return;
            }
        }
        panic!("never reached sustain stage");
    }

    #[test]
    fn release_starts_from_current_level() {
        let mut env = Envelope {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.5,
            release: 0.1,
            ..Envelope::default()
        };
        env.stage = EnvStage::Attack;
        env.time = 0.0;
        env.level = 0.0;

        let dt = 1.0 / 48000.0;
        for _ in 0..1000 {
            env.tick(dt);
        }
        assert_eq!(env.stage, EnvStage::Decay);
        let level_at_gate_off = env.level;
        assert!(
            level_at_gate_off > 0.5 && level_at_gate_off < 1.0,
            "should be in decay, level={level_at_gate_off}"
        );

        env.release_start_level = env.level;
        env.stage = EnvStage::Release;
        env.time = 0.0;
        let first_release = env.tick(dt);
        assert!(
            (first_release - level_at_gate_off).abs() < 0.01,
            "release should start from {level_at_gate_off}, got {first_release}"
        );
    }

    #[test]
    fn zero_attack_time_instant() {
        let mut env = Envelope {
            attack: 0.0,
            decay: 0.1,
            sustain: 0.7,
            release: 0.3,
            ..Envelope::default()
        };
        env.stage = EnvStage::Attack;
        env.time = 0.0;
        env.level = 0.0;

        let val = env.tick(1.0 / 48000.0);
        assert!(
            (val - 1.0).abs() < 1e-6,
            "zero attack should reach 1.0 on first tick, got {val}"
        );
    }

    #[test]
    fn curve_linear_when_near_zero() {
        assert!((curve(0.5, 0.0) - 0.5).abs() < 0.01);
        assert!((curve(0.25, 0.001) - 0.25).abs() < 0.01);
    }
}
