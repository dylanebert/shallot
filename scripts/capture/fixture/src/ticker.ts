import { type Plugin, type State, type System, sparse, u32 } from "@dylanebert/shallot";

// The hot-reload capture fixture (scripts/capture/flows/hot-reload.pw.ts). `ticks` is the runtime
// counter that climbs every frame and must SURVIVE an in-place swap — a rebuild would reset it, so it's
// the proof that the live State was patched, not torn down. `mark` is the behavior output the flow edits:
// the system writes the MARK constant below each frame, so flipping MARK is a code edit that changes
// `mark` live while `ticks` keeps climbing. The system runs `mode: "always"` so it ticks in the editor's
// edit mode (a plain "play" system is skipped there), and it only sets field values — non-destructive,
// per the edit-mode contract.
export const Ticker = { ticks: sparse(u32), mark: sparse(u32) };

const MARK = 1;

const TickSystem: System = {
    name: "tick",
    group: "simulation",
    annotations: { mode: "always" },
    update: (state: State) => {
        for (const eid of state.query([Ticker])) {
            Ticker.ticks.set(eid, Ticker.ticks.get(eid) + 1);
            Ticker.mark.set(eid, MARK);
        }
    },
};

export const TickerPlugin: Plugin = {
    name: "ticker",
    components: { Ticker },
    systems: [TickSystem],
    // warm-spawned, not scene-authored: the entity lives in State, so a swap (which doesn't re-run warm)
    // preserves it and its accumulated ticks, while a rebuild re-runs warm and resets the counter.
    warm: (state) => {
        const eid = state.create();
        state.add(eid, Ticker);
    },
};

// default-exported so `shallot.json` can declare it by path (`"ticker": "./src/ticker"`)
export default TickerPlugin;
