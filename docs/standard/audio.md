---
title: Audio
description: sound, music, spatial audio
source: standard/audio
icon: audio-lines
---

# Audio

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

Shallot's audio is a procedural synth engine with a WASM kernel. Three layers build on each other:

- **Instruments** define sound as a declarative DAG of typed nodes (oscillators, filters, envelopes, samples, modulation routing). `instrument()` compiles the DAG into a flat eval order. Wavetable and PolyBLEP oscillators, state-variable filter, ADSR envelopes, PCM sample playback
- **Patterns** are MIDI-shaped note sequences. Tracks map instrument names to sorted note arrays. `pattern()` compiles them for playback
- **Audio state** is a plain data interface (`AudioState`) with a 64-voice pool, generation counters, 8 independent transports, spatial and acoustic batching. Pure functions operate on it. The backend talks to an `AudioWorklet` via batched `postMessage`

ECS integration sits on top. The `Sound` component means "playing" — add it to start, remove to stop. `Listener` marks the spatial listener entity. Both paths (ECS-managed and direct `audio/core` functions) share the same voice pool.

Lower layers never import from higher ones. `engine.ts` has no pattern knowledge. Direct voice control bypasses ECS entirely.

## Examples

### Playing a sound

End-to-end: load an audio file, wrap it as an instrument, reference it from the scene or spawn at runtime.

```typescript
import { sample, instrument } from "@dylanebert/shallot";

const sid = sample("/boom-hit.mp3", "boom");
instrument({
    nodes: { src: { type: "sample" } },
    output: "src",
    values: { "src.bufferId": sid, "src.rate": 1, "src.loop": 0, "src.volume": 1 },
}, "boom");
```

Run this once at app startup, before the scene loads. The instrument is now registered under the name `"boom"`.

Reference it from scene XML on a long-lived entity (ambient/loop):

```xml
<a id="ambient" sound="instrument: boom; loop: 1; volume: 0.4" transform />
<a id="player" camera listener transform="pos: 0 1.6 0" />
```

Or spawn at runtime for one-shots (e.g. tile placement, impact):

```typescript
import { Sound, Transform } from "@dylanebert/shallot";
import { instrumentRegistry } from "@dylanebert/shallot/audio/core";

function playAt(state: State, name: string, x: number, y: number, z: number) {
    const eid = state.addEntity();
    state.addComponent(eid, Transform);
    state.addComponent(eid, Sound);
    Transform.posX[eid] = x; Transform.posY[eid] = y; Transform.posZ[eid] = z;
    Sound.instrument[eid] = instrumentRegistry.getByName(name)!;
    Sound.spatial[eid] = 1;
}
```

`Sound.loop = 0` (the default) makes it a one-shot — SoundSystem destroys the entity when the envelope idles. Spatial routing requires a `Listener` in the scene; without one, voices play direct stereo.

The `instrument()` call is the procedural sound design surface — graphs of oscillators, filters, envelopes, samples, and modulation. The example above is the trivial case (one sample node); verbosity is the cost of the graph being explicit. The workstation example demonstrates richer instruments. Beginner-friendly asset management is a separate concern, handled by the editor when it matures.

### Direct playback

Import functions from `audio/core` and operate on the `AudioState` resource:

```typescript
import { Audio, allocTransport, setBPM, alloc, assign, schedule, play, beat } from "@dylanebert/shallot/audio/core";

const audio = Audio.from(state)!;
const tid = allocTransport(audio);
setBPM(audio, tid, 120);

const slot = alloc(audio);
assign(audio, slot, instrumentId);
schedule(audio, tid, beat, slot, duration, params);
play(audio, tid);

// poll beat(audio, tid) for loop detection
// stop: gate off, free voices, freeTransport(audio, tid)
```

### One-shots (direct API)

Alloc a voice, assign an instrument, set params, gate on. Register `onIdle` to free the voice when the envelope finishes. The convolver tail flushes automatically before the voice is freed.

### Spatial audio

`state.only([Listener, WorldTransform])` finds the listener entity. SoundSystem handles spatial audio automatically for voices with `Sound.spatial = 1`. For direct use, `polar()` (from `audio/core`) computes azimuth/elevation/distance, then `addSpatial(audio, ...)` + `flushSpatial(audio)` once per frame. No listener means voices play direct stereo.

Non-spatial voices skip FOA+HRTF and write directly to the stereo bus (`setVoiceSpatial(audio, slot, false)`).

### ECS param upload

SoundSystem uploads volume and pitch every frame for active voices. Instruments declare `volumeParam` and `pitchParams` at registration — compiled into param offsets on voice allocation.

- **Volume:** `Sound.volume[eid]` (default 1.0) applies a quadratic curve (`v * v`), multiplied by the instrument's base `vol.level`. Linear 0–1 knob maps to -∞ to 0 dB
- **Pitch:** iterates all `pitchParams` entries. Each gets `noteFreq(baseFreq, octave, Sound.pitch[eid] + semitone, fine)`. Multi-oscillator instruments list all osc frequency params for detuning/layering

### Gain staging

- All oscillator waveforms peak at ±1.0 (peak normalization, not RMS)
- Volume curve is quadratic: 50% knob = -12 dB
- Convolver IRs are energy-normalized, with extracted energy driving per-voice `refl_gain` (capped at 1.0)
- Master output uses `tanh()` soft limiter — signals below ~0.7 pass unchanged, gentle saturation above

### Sample playback

`sample(source, name?)` accepts:

- a URL string — fetched and decoded in the background (mp3, wav, ogg, flac)
- a `Blob` or `File` — decoded in the background
- a `Float32Array` — registered immediately as raw PCM (procedural / synthesized content)

The id returns synchronously in all cases. Stereo files are downmixed to mono. Playing before decode completes plays silence — call `whenLoaded(id)` to await readiness, or `getSample(id)` to read back the decoded buffer and sample rate.

Sample node params: `bufferId` (sample registry id, discrete), `rate` (1.0 = unit playback, 2.0 = octave up), `loop` (0/1 discrete), `volume`. Position resets to 0 on instrument assignment. One-shot samples produce silence past the end; looping samples wrap. Compose with envelopes and filters by routing the Sample node's output through them like any other source.

### Suspended AudioContext

When `AudioContext` is suspended (page loaded without user interaction), `running(audio)` is false. SoundSystem discards pending one-shot entities to prevent burst playback on resume. Looping sounds stay pending.


<!-- tab: Reference -->

<!-- API:standard/audio -->

<!-- CORE:audio -->

<!-- /tabs -->
