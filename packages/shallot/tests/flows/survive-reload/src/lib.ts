import {
    type Config,
    OrbitPlugin,
    type Plugin,
    type State,
    type System,
    sparse,
    u32,
} from "@dylanebert/shallot";

// The survive-reload flow's standalone app — the real production reload path a bun test can't reach.
// `Counter.n` climbs every frame: it's the runtime value the reload must bring back (a fresh build starts
// at 0). The `hero` counter is scene-authored, so it's in load's authored set and `serialize` captures it;
// `Sprout` is warm-derived (absent from the authored set), so a restore re-derives exactly one, never doubled.
export const Counter = { n: sparse(u32) };
export const Sprout = {};

const TickSystem: System = {
    name: "survive-tick",
    group: "simulation",
    update: (state: State) => {
        for (const eid of state.query([Counter])) Counter.n.set(eid, Counter.n.get(eid) + 1);
    },
};

const SurvivePlugin: Plugin = {
    name: "survive",
    components: { Counter, Sprout },
    traits: { Counter: { defaults: () => ({ n: 0 }) } },
    systems: [TickSystem],
    // one Sprout per authored Counter, re-derived on every build — the not-doubled check
    warm: (state) => {
        for (const _ of state.query([Counter])) {
            const sprout = state.create();
            state.add(sprout, Sprout);
        }
    },
};

// inline scene, never a public/scenes file: a .scene on disk makes `shallot verify` classify the dir as a
// manifest project, whose synthesized entry bypasses this app's index.html + main.ts. Inline XML is also
// the path the restore takes, so both boots load through the same codepath.
const scene = `<scene>
    <a ambient-light="intensity: 0.6" />
    <a directional-light="direction: -0.4 -1 -0.55; color: 0xfff4e0; intensity: 1.2" />

    <a camera sear orbit="distance: 5; yaw: 0.5; pitch: 0.35" transform />

    <!-- the authored entity whose runtime-climbing counter.n the reload must restore -->
    <a id="hero" part counter transform color="rgba: 0.85 0.55 0.35" />
</scene>`;

export const config: Config = {
    plugins: [OrbitPlugin, SurvivePlugin],
    scene,
};
