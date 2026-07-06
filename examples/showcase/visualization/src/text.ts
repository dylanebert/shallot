import {
    f32,
    type Plugin,
    type System,
    sparse,
    TextPlugin,
    Transform,
    TransformsPlugin,
    TweenPlugin,
} from "@dylanebert/shallot";
import { start } from "./boot";

// SDF text labels on the kitchen renderer, authored in the scene. The headline behavior: positions
// animate (the title bobs on a declarative tween, the captions orbit via this system) while the glyph
// buffer lays out once and never rebuilds — motion flows through the Transform slab the VS samples, not
// the glyph geometry. `Ring` marks an orbiting caption and carries its phase offset; that orbit (coupled
// cos/sin on two axes) is the genuinely bespoke motion, so it's a system, not a tween.
const Ring = { phase: sparse(f32) };

const RingSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },
    update(state) {
        const t = performance.now() * 0.001;
        for (const eid of state.query([Ring, Transform])) {
            const a = t * 0.5 + Ring.phase.get(eid);
            Transform.pos.set(eid, Math.cos(a) * 3.4, Math.sin(a + t) * 0.4, Math.sin(a) * 3.4, 0);
        }
    },
};

const RingPlugin: Plugin = {
    name: "TextRing",
    components: { Ring },
    systems: [RingSystem],
    traits: { Ring: { requires: [Transform], defaults: () => ({ phase: 0 }) } },
    dependencies: [TransformsPlugin],
};

await start([TextPlugin, TweenPlugin, RingPlugin], "../scenes/text.scene");
