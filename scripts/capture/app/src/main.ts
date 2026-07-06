import { run, serialize, stringify } from "@dylanebert/shallot";
import { Counter, config, Sprout } from "./lib";

// Survive a browser reload, composed from the engine's atomic primitives — no engine flag. The core is
// already sufficient: `serialize(state)` snapshots the live State, `run({ scene })` loads it back through
// the same inline-XML path a fresh scene takes (`warm` re-derives the rest). Restore on boot, re-snapshot
// on unload. The app owns the policy — storage backend, trigger, key — not the engine.
const KEY = "shallot:capture-survive";
const saved = sessionStorage.getItem(KEY);
const { state, dispose } = await run(saved ? { ...config, scene: saved } : config);

const persist = () => sessionStorage.setItem(KEY, stringify(serialize(state)));
addEventListener("beforeunload", persist);

// the read seam the survive-reload flow polls: the max climbing counter + the warm-derived sprout count.
// re-set on every boot (a real reload re-runs this module against the restored State).
(window as Window & { __survive?: () => { n: number; sprouts: number } }).__survive = () => {
    let n = 0;
    for (const eid of state.query([Counter])) n = Math.max(n, Counter.n.get(eid));
    return { n, sprouts: [...state.query([Sprout])].length };
};

// HMR re-runs this module — drop the listener and dispose the old State + RAF loop, or each reload stacks another.
if (import.meta.hot) {
    import.meta.hot.dispose(() => {
        removeEventListener("beforeunload", persist);
        dispose();
    });
}
