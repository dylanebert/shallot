import {
    type Plugin,
    Sound,
    type State,
    type System,
    sample,
    Transform,
} from "@dylanebert/shallot";

// `"Audio": true` enables the worklet + voice allocator; a `listener` on the camera is the spatial
// reference. A `sound` on an entity plays a registered instrument by name; giving that entity a
// `transform` makes the voice positional — pan and distance track it relative to the listener.
const SR = 48000;

// a seamless sine tone — a whole number of cycles per buffer, so the loop has no click at the seam
function tone(freq: number, seconds: number): Float32Array {
    const out = new Float32Array(Math.round(seconds * SR));
    for (let i = 0; i < out.length; i++) out[i] = Math.sin((2 * Math.PI * freq * i) / SR) * 0.6;
    return out;
}

// derive position from State each frame (no module-level accumulator) so the orbit survives a rebuild
const OrbitSource: System = {
    name: "audio-orbit-source",
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([Sound, Transform])) {
            if (Sound.loop.get(eid) === 1) {
                Transform.pos.set(eid, Math.cos(t * 0.7) * 4, 0, Math.sin(t * 0.7) * 4, 0);
            }
        }
    },
};

const AudioDemo = {
    name: "AudioDemo",
    systems: [OrbitSource],
    initialize() {
        sample(tone(240, 1), "tone"); // register before scene parse so `instrument: tone` resolves
    },
} satisfies Plugin;

// spawn one-shots at runtime with `play(state, name, { pos })`; cap a rapidly-retriggered sound with
// `sfx(name, { max, cooldown })`
export default AudioDemo;
