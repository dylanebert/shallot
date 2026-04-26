---
paths:
    - "packages/shallot/src/standard/audio/**/*.ts"
    - "packages/shallot/rust/audio/**/*.rs"
    - "packages/shallot/src/extras/acoustics/**/*.ts"
    - "examples/workstation/**/*"
---

# Audio

Reference: `docs/standard/audio.md` for architecture, playback patterns, and API surface.

## Voice lifecycle

- **One-shot voices** (`one_shot` flag): auto-gate-off when any envelope reaches Sustain. Sustain level becomes release start. Prevents voice accumulation from short-lived sounds.
- **Spatial tail**: spatial voices get min 16 blocks (~46ms) idle tail, regardless of convolver state. Covers GPU reflection readback latency.
- **Idle tracking**: worklet's `_releasing` set watches for idle. `voice_idle` returns true when all envelopes are `Idle` AND idle_countdown reaches 0. Only `watch_idle` populates `_releasing` — never `gate(0)`. Gate-off is musical; idle watching is lifecycle.
- **Transport re-gate**: transport events gate on/off internally without notifying the worklet. If `gate(0)` populated `_releasing`, a voice gated off by seek then re-gated would be falsely reported idle.

## Two update paths

- **Value change** (knob tweak): `setValues()` + `upload(audio, instId, changed)`. No recompilation, no voice reset
- **Topology change** (add/remove nodes): `instrument()` to recompile + `refresh(audio, instId)`. Sends new topology to voices

Don't recompile for value-only changes.

## Compiled topology vs mutable values

`CompiledInstrument` is immutable compiled topology (nodes, paramLayout, modulations, outputBuf). Values stored separately in instrument.ts.

- `instrumentRegistry.get(id)` — immutable topology
- `getValues(id)` — mutable values map (metadata + WASM params)
- `getParamPairs(instId)` — offset/value pairs filtered to paramLayout (for WASM upload)
- `setValues(instId, params)` — mutates the values map

All accessed via `@dylanebert/shallot/audio/core` subpath. User-facing API is `instrument()`, `Instrument` type, `noteFreq`, `midiFreq` from the main barrel.

## Instrument completeness

`instrument()` sets a default for every `paramLayout` key. A completeness assertion runs after — missing keys throw. Never silently fall back (`?? 0`, `k.default`). Missing keys are bugs.

## Metadata params vs WASM params

Values contain two kinds: **WASM params** (in `paramLayout`, sent to kernel) and **metadata params** (global `octave`/`semitone`/`fine`/`volume` + per-oscillator variants, NOT in `paramLayout`, never sent to WASM). `getParamPairs()` automatically excludes metadata. Frequency calculation belongs in JS where note/tuning context lives.

When adding new per-instrument values: if the kernel needs it, add to a node type's param list. If only JS needs it, add to metadata defaults in `instrument()`.

## Gain staging

- Histogram readback normalized by `NUM_RAYS` only (GPU shader includes 1/(4π) in distance attenuation). Per-frame clear, no accumulation. EMA smoothing (alpha=0.3) in `processHistogram` handles noise reduction across ~10Hz dispatches
- Convolver IRs energy-normalized in `reconstructIR`, extracted energy drives `refl_gain` per voice (capped at 1.0)
- `refl_gain` initializes to 0.0 on voice activation — reflections fade in as acoustics data arrives
- All block-rate parameters (`gain`, `refl_gain`, FDN `wet_gain`/`eq`, FOA encode coefficients) interpolated per-sample across each block to prevent zipper noise (ref: Steam Audio `gain_effect.cpp`)
- Occlusion affects direct path only — convolver and FDN process the unoccluded signal (ref: Steam Audio `direct_effect.cpp`)
- Occlusion and transmission are separate: `gain = occlusion + (1 - occlusion) × transmission` per band (ref: Steam Audio `direct_effect.cpp`). Volumetric occlusion (8 rays to source sphere) for smooth transitions near edges

## Gotchas

- Filter `mix` param controls wet/dry. At `mix=0` the filter is fully bypassed. Always set `"filter.mix": 1` when active
- DSP hot path never crosses FFI mid-block — all synthesis at ~375 Hz in `process()`
- Transport events persist (cursor-based, not consumed). `transport_stop` resets position/cursor but preserves events. `transport_clear_events` removes them
- `transport_seek` gates off all voices with events on that transport (enters Release)
