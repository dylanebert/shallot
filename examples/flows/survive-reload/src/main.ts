import { run, serialize, stringify } from "@dylanebert/shallot";
import { installHarness } from "@dylanebert/shallot/harness";
import { Counter, config, Sprout } from "./lib";

// The app self-drives the reload dance `shallot verify` observes. Survival is composed from the engine's
// atomic primitives — no engine flag: `serialize(state)` snapshots the live State, `run({ scene })` loads it
// back through the same inline-XML path a fresh scene takes (`warm` re-derives the rest). The real path stays
// real: a `beforeunload` listener serializes to sessionStorage, and boot restores from it in place of the
// scene. First boot climbs then `location.reload()`s; the restored boot installs the published harness and
// its `run()` asserts the runtime value survived and the warm-derived entity wasn't doubled.

const STATE_KEY = "shallot:flow-survive:state";
const MARK_KEY = "shallot:flow-survive:mark";
const N_KEY = "shallot:flow-survive:n";

// a never-ready placeholder pins verify to the harness path from the first synchronous statement of
// module evaluation — before `await run()`, on both boots. Set after the await, it loses a race: the
// unified wait polls ~500ms apart (wider over the wsl-bridge), the whole placeholder window (run
// resolves → ~21 climb frames → reload) fits between two polls, and verify then settles on the
// restored boot's static frame as a bare smoke — never seeing the dance. The restored boot's
// installHarness replaces this with the real, ready harness.
window.__harness = { ready: false };

const restoring = sessionStorage.getItem(MARK_KEY) !== null;
const saved = sessionStorage.getItem(STATE_KEY);
const { state, dispose } = await run(restoring && saved ? { ...config, scene: saved } : config);

// the real production path: serialize the live State to sessionStorage on unload; a reload restores it.
const persist = () => sessionStorage.setItem(STATE_KEY, stringify(serialize(state)));
addEventListener("beforeunload", persist);

// HMR re-runs this module — drop the listener and dispose the old State + RAF loop, or each edit stacks another.
if (import.meta.hot) {
    import.meta.hot.dispose(() => {
        removeEventListener("beforeunload", persist);
        dispose();
    });
}

const maxN = (): number => {
    let n = 0;
    for (const eid of state.query([Counter])) n = Math.max(n, Counter.n.get(eid));
    return n;
};
const sproutCount = (): number => [...state.query([Sprout])].length;

if (!restoring) {
    // first boot: let Counter.n climb well past the scene default (0), then mark + reload. `beforeunload`
    // serializes the live State on the way out — the path the restore exercises.
    const climb = (): void => {
        if (maxN() > 20) {
            sessionStorage.setItem(N_KEY, String(maxN()));
            sessionStorage.setItem(MARK_KEY, "1");
            location.reload();
            return;
        }
        requestAnimationFrame(climb);
    };
    requestAnimationFrame(climb);
} else {
    // restored boot: the counter came back from the serialized snapshot and kept climbing, so n >= stored;
    // the warm-derived sprout re-derived exactly once. Install the published harness and report it.
    const stored = Number(sessionStorage.getItem(N_KEY) ?? "0");
    const harness = installHarness(state);
    harness.run = async () => {
        const n = maxN();
        const sprouts = sproutCount();
        const nOk = n >= stored && stored > 20;
        const sproutOk = sprouts === 1;
        return {
            ok: nOk && sproutOk,
            checks: [
                {
                    name: "runtime value survived the reload",
                    ok: nOk,
                    detail: `n=${n} (stored ${stored})`,
                },
                {
                    name: "warm-derived sprout not doubled",
                    ok: sproutOk,
                    detail: `sprouts=${sprouts}`,
                },
            ],
            n,
            stored,
            sprouts,
        };
    };
}
