---
paths:
    - "packages/shallot/src/standard/audio/**/*.ts"
    - "packages/shallot/rust/audio/**/*.rs"
---

# Audio

> The **kernel-behavior** sections below (voice lifecycle, instrument completeness, metadata params, gain staging, gotchas) describe the stable `rust/audio/` kernel + `worklet.ts`. The **Layers** section describes the live `standard/audio` JS layer.
>
> The kernel carries the full standard insert rack (delay, dynamics, waveshaper, EQ, chorus, flanger, phaser, tremolo) alongside the synth primitives — every effect transcribed from a permissive reference and bit-checked against `src/golden.rs`. Adding a node is a real reopening, not routine: a new `NodeType`, its `graph.rs` dispatch + `NodeState`/`delay_lines` init in `set_voice_instrument` (lib.rs), a golden vector, a behavioral test, and the `instrument.ts` table rows. Effect DSP is a few floats inline in the Copy `NodeState`; sample-buffer nodes (delay, chorus, flanger) box a `DelayLine` in the `Voice.delay_lines` side-array keyed by node index, allocated on the control path like `Voice.convolver`.

## Layers

Substrate then composable contracts, dependencies inward (the `render/` ← `sear/` shape). Don't collapse layers; don't push an upper layer's work into a lower one.

- **`audio/core` — the voice contract + DSP substrate.** The `Audio` singleton (AudioContext/worklet host, 64-slot voice allocator, per-frame batch), the gen-validated voice ops, `polar` + the spatial batch, and the DAG compiler (`instrument()` → kernel node-list + `paramLayout`). The kernel owns all DSP; this owns allocation + the wire — no CPU voice mirror, no `AudioCommand` union (the wire lives only in `worklet.ts`).
- **`audio/index` — the SFX surface (the barrel happy path).** `Sound`/`Listener` on `sparse`, `SoundSystem` (alloc/free gated on the `Voiced` marker, volume firehose, spatial derivation), `play()`.
- **`audio/policy` — the per-name instance budget over `play()` (FMOD/Wwise instance limits).** `sfx(name, {max, cooldown, steal})` registers a limit `play()` consults (cap / min-interval cooldown / cull victim oldest|quietest|drop); no policy → unbounded as before. Cooldown is clocked on `state.time.elapsed`, self-healing across a rebuild via a backwards-clock read. **Loops are never culled** — `steal()` skips `loop === 1` and a per-name cap removes only a same-name instance, so an SFX burst can't steal the bed/music. A general `Sound.priority` is the deferred follow-up (a real scope boundary, not a stopgap).
- **Future contracts on top** (each one scenario, mix freely): pattern/sequencer (the transport L2 wires aren't built — a pure L3 + thin-passthrough add), the spatial/acoustics rebuild, physics coupling, DDSP. Substrate stays small; use cases accumulate at the top.

## Voice lifecycle

- **One-shot voices** (`one_shot` flag): auto-gate-off when any envelope reaches Sustain. Sustain level becomes release start. Prevents voice accumulation from short-lived sounds.
- **Spatial tail**: spatial voices get min 16 blocks (~46ms) idle tail, regardless of convolver state. Covers GPU reflection readback latency.
- **Idle tracking**: worklet's `_releasing` set watches for idle. `voice_idle` returns true when all envelopes are `Idle` AND idle_countdown reaches 0. Only `watch_idle` populates `_releasing`; any `gate(value != 0)` clears it (the re-gate guard below), so a voice's `watch_idle` must be enqueued **after** its gate-on or the same batch wipes the watch and the voice never idles (it leaks). Gate-off is musical; idle watching is lifecycle.
- **Sample playback path**: a one-shot rides `play(name)` → `sampler()` (the enveloped auto-build). With no envelope `voice_idle` is vacuously true, so a bare sample node is freed mid-sample by `watchIdle`; the sampler's sustain=1/decay≈length envelope holds it open and also wires `volumeParam` (else `Sound.volume` is ignored) and `loopParam`. A **looping** bed uses an explicit looping instrument with `volumeParam` — the auto-sampler's loop arm is unexercised in-engine.
- **Transport re-gate**: transport events gate on/off internally without notifying the worklet. If `gate(0)` populated `_releasing`, a voice gated off by seek then re-gated would be falsely reported idle.

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
