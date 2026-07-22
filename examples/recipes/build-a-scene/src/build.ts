import { Color, Part, type Plugin, type State, Transform } from "@dylanebert/shallot";

// A scene composes the world declaratively — see public/scenes/build-a-scene.scene, where each `<a>` is
// an entity and each attribute a component. The same world can be built from code: `state.create` mints
// an entity, `state.add` gives it a component, and the component's fields set its data. Procedural
// generation and tests build this way; hand-authored content stays in the scene file.
//
// Spawn in `warm` (after the scene parses) so these derived entities re-create on every rebuild and are
// never double-serialized. Here a ring of parts surrounds the authored scene.
const Ring = {
    name: "Ring",
    warm(state: State) {
        const count = 8;
        for (let i = 0; i < count; i++) {
            const a = (i / count) * Math.PI * 2;
            const eid = state.create();
            state.add(eid, Transform);
            state.add(eid, Part);
            state.add(eid, Color);
            Transform.pos.set(eid, Math.cos(a) * 3.5, 0.4, Math.sin(a) * 3.5, 0);
            Transform.scale.set(eid, 0.4, 0.4, 0.4, 0);
            Color.rgba.set(eid, 0.5 + 0.4 * Math.cos(a), 0.55, 0.5 + 0.4 * Math.sin(a), 1);
        }
    },
} satisfies Plugin;

export default Ring;
