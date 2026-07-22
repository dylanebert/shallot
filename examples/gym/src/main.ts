import "./scenarios";
import {
    frames,
    getScenario,
    installHarness,
    mountControls,
    resolveParams,
    scenarioNames,
} from "./gym";

// One app, one URL-selected scenario, one `build`. `?scenario=` picks it; with no scenario the page is
// an index of every registered name. Each scenario declares its tunables as `params` (the single source
// of truth): the URL parses them, the bench `--param` sets them, and the top-right control panel
// auto-renders from them — a live knob mutates in place, a structural (`rebuild`) knob reloads. The same
// page runs headless under the harness or interactive in a tab; `build` attaches the camera either way,
// the `#hud` shows the live readout, and F3 toggles the profiler stats panel. No environment switch.

const url = new URL(window.location.href);
const name = url.searchParams.get("scenario");

if (name === null) {
    renderIndex();
} else {
    await boot(name);
}

// no scenario selected — list every registered scenario as a link into its page.
function renderIndex(): void {
    document.getElementById("panel")?.remove();
    const index = document.createElement("div");
    index.id = "index";
    index.innerHTML = "<h1>gym</h1><p>scenarios</p>";
    const nav = document.createElement("nav");
    for (const n of [...scenarioNames()].sort()) {
        const link = document.createElement("a");
        link.href = `?scenario=${n}`;
        link.textContent = n;
        nav.appendChild(link);
    }
    index.appendChild(nav);
    document.getElementById("app")!.appendChild(index);
}

async function boot(name: string): Promise<void> {
    const scenario = getScenario(name);
    if (!scenario) {
        throw new Error(`unknown scenario "${name}". Available: ${scenarioNames().join(", ")}`);
    }

    const decls = scenario.params ?? [];
    const values = resolveParams(decls, url.searchParams);

    // a never-ready placeholder pins `shallot verify` to the harness path before the build starts: a
    // heavy scenario (a large pile) builds for longer than the settle check needs to conclude, and a
    // settle verdict would skip the scenario's checks entirely. installHarness replaces it once built.
    window.__harness = { ready: false };

    const canvas = document.createElement("canvas");
    document.getElementById("app")!.appendChild(canvas);

    const { state, dispose } = await scenario.build(canvas, values);

    let built = false;
    installHarness(scenario, state, () => built);
    await frames(2);
    built = true;

    // a control change writes the value into the URL (so a reload restores it), then either reloads for a
    // structural knob (clean rebuild from the new scene size/shape) or lets a live knob take effect next
    // frame — the scenario reads `values` each frame, so a live mutation needs no rebuild.
    const panel = document.getElementById("panel")!;
    const controlsCleanup = mountControls(panel, decls, values, (key, rebuild) => {
        const v = values[key];
        const u = new URL(window.location.href);
        u.searchParams.set(key, typeof v === "boolean" ? (v ? "1" : "0") : String(v));
        history.replaceState(null, "", u);
        if (rebuild) location.reload();
    });
    state.onDispose(controlsCleanup);

    let hudFrame = 0;
    const hud = document.getElementById("hud")!;
    if (scenario.live) {
        const update = () => {
            hud.textContent = scenario.live!(state);
            hudFrame = requestAnimationFrame(update);
        };
        hudFrame = requestAnimationFrame(update);
        state.onDispose(() => cancelAnimationFrame(hudFrame));
    }

    // HMR re-runs this module — `dispose()` disposes the State, unwinding the controls + HUD loop
    // registered above (`state.onDispose`), so each reload starts clean without stacking another loop.
    if (import.meta.hot) {
        import.meta.hot.dispose(dispose);
    }
}
