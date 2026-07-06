import {
    type Plugin,
    Sound,
    type State,
    type System,
    sample,
    Transform,
} from "@dylanebert/shallot";

// #doc:intro
// Spatial procedural audio: a looping tone orbits the listener, its pan and volume tracking its position.

// #doc:code source:audio/public/scenes/audio.scene
// `"Audio": true` enables the worklet and voice allocator. A `listener` on the camera makes it the spatial
// reference; a `sound` on an entity plays an instrument by name, and giving that entity a `transform` makes
// the voice positional — pan and distance are relative to the listener.

// #doc:code
// ### Register the sound, move the source
//
// An instrument or sample registers by name before the scene parses, so `sound="instrument: tone"`
// resolves. Here a seamless sine buffer registers as "tone", and a system orbits the source each frame so
// its spatial pan audibly tracks its position:
// #region demo
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
// #endregion

// #doc:code
// Spawn one-shots at runtime with `play(state, name, { pos })` instead of authoring them in the scene, and
// cap a rapidly-retriggered sound with `sfx(name, { max, cooldown })`.

export default AudioDemo;
