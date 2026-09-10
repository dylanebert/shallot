---
paths:
  - "packages/{shallot/src/standard/audio/**/*.ts,shallot/rust/audio/**/*.rs}"
---

# Audio

Kernel behavior lives in `rust/audio/` + `worklet.ts`; JS layers in `standard/audio`. DSP nodes come from permissive references, bit-checked against `src/golden.rs`. A new node reopens the contract: `NodeType`, graph dispatch, `NodeState`/delay storage initialization in `set_voice_instrument`, golden vector, behavioral test and `instrument.ts` rows. Inline Copy state holds small DSP; sample buffers use boxed `DelayLine` in the voice's node-keyed side-array, allocated on control path like the convolver.

## Layers

Dependencies inward, substrate then composable contracts; don't collapse layers or push upper work down. Core owns AudioContext/worklet hosting, 64-slot generation-validated allocation, per-frame wire/spatial batches and DAG compilation to nodes/`paramLayout`. Kernel owns all DSP; no CPU voice mirror or duplicate command union: wire belongs only to `worklet.ts`.

Index: SFX, sparse Sound/Listener, SoundSystem alloc/free gated by Voiced, volume/spatial updates/play. Policy registers per-name max/cooldown/oldest|quietest|drop limits over play; no policy means unbounded. Cooldown uses `state.time.elapsed`, heals backwards clocks after rebuild. Never cull loops; caps steal only same-name voices, not beds/music. General priority is out of scope. Future sequencer/acoustics/physics/DDSP contracts: one scenario each, mix freely above small substrate; sequencer transport wires aren't built.

## Voice lifecycle

One-shots auto-gate-off when any envelope reaches Sustain; sustain level starts Release. Spatial voices retain at least 16 idle blocks (~46ms), regardless of convolver, for reflection-readback latency.

`voice_idle` requires all envelopes Idle AND countdown zero. Only `watch_idle` populates worklet `_releasing`; every nonzero gate clears it, so enqueue watch AFTER gate-on even within one batch. Gate-off is musical, not idle watching: transport gates internally, and seek-off/re-gate must not falsely free voices.

One-shot samples use play→sampler's enveloped auto-build: bare samples have vacuously idle envelopes and free mid-sample. Sustain 1/decay≈length holds playback; wire `volumeParam` and `loopParam` or volume is ignored. Looping beds need explicit looping instruments with volume wiring; auto-sampler loop remains unexercised in-engine.

## Instrument completeness

`instrument()` supplies EVERY `paramLayout` default, then asserts completeness; missing keys throw, never fallback to zero or a kernel default. Kernel-needed values belong to node param lists; JS-only values to metadata defaults.

WASM params alone enter `paramLayout`/kernel uploads. Global/per-oscillator octave/semitone/fine/volume metadata never goes to WASM; `getParamPairs()` excludes it. Frequency calculation stays in JS with note/tuning context.

## Gain staging

Activation sets `refl_gain` to zero; fade reflections in as acoustics arrives. Interpolate ALL block-rate gain/reflection/FDN wet-gain/EQ/FOA coefficients per sample to avoid zipper noise (Steam Audio `gain_effect.cpp`).

Occlusion affects direct path only; convolver/FDN receive unoccluded signal. Occlusion and transmission stay separate: per-band gain = occlusion + (1 − occlusion) × transmission. Volumetric occlusion uses 8 rays to source sphere for smooth edges (Steam Audio `direct_effect.cpp`).

## Gotchas

Filter mix 0 bypasses; active filters set `"filter.mix": 1`. No mid-block FFI; synth runs in process at ~375Hz. Transport events persist by cursor: stop resets position/cursor, not events; clear removes events. Seek gates off all voices with events on that transport into Release.
