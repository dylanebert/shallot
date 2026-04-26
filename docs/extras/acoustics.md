---
title: Acoustics
description: realistic sound, echoes, occlusion
source: extras/acoustics
icon: audio-waveform
---

# Acoustics

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

`AcousticPlugin` depends on `PhysicsPlugin` — no physics means no acoustics (sounds play dry). Three GPU-driven layers using the physics LBVH:

- **Occlusion** — shadow rays per spatial source with `Acoustic` marker. Binary hit/miss per-voice gain + LPF cutoff
- **Reflections** — Monte Carlo ray bouncing from listener (4096 rays, 16 bounces, ~10Hz). Energy accumulated into per-source histograms across dispatches, converted to impulse responses for per-voice FFT convolvers. Late energy RT60 estimation for global FDN reverb
- **Materials** — per-body `AcousticMaterial` component, packed and uploaded directly to GPU


<!-- tab: Reference -->

<!-- API:extras/acoustics -->

<!-- /tabs -->
