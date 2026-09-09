import "./scenarios";
import { frames, getScenario, installHarness, resolveParams, scenarioNames } from "./gym";

// The bench fixture page: one URL-selected scenario, one `build`. `?scenario=` picks it; the scenario's
// `params` are the single source of truth the URL parses and `bun bench --param` sets. `shallot verify`
// boots this page — `bun bench` is a wrapper over it — so this is the harness fixture, not an app: there
// is no index, no control panel and no HUD. The scenario itself never branches on how it was loaded.

const url = new URL(window.location.href);
const name = url.searchParams.get("scenario");
if (name === null) throw new Error(`no ?scenario= given. Available: ${scenarioNames().join(", ")}`);

const scenario = getScenario(name);
if (!scenario) {
    throw new Error(`unknown scenario "${name}". Available: ${scenarioNames().join(", ")}`);
}

const values = resolveParams(scenario.params ?? [], url.searchParams);

// a never-ready placeholder pins `shallot verify` to the harness path before the build starts: a
// heavy scenario (a large pile) builds for longer than the settle check needs to conclude, and a
// settle verdict would skip the scenario's checks entirely. installHarness replaces it once built.
window.__harness = { ready: false };

const canvas = document.createElement("canvas");
document.getElementById("app")!.appendChild(canvas);

const { state, dispose } = await scenario.build(canvas, values);

let built = false;
installHarness(scenario, state, () => built, values);
await frames(2);
built = true;

// HMR re-runs this module — dispose the old State + RAF loop, or each edit stacks another.
if (import.meta.hot) {
    import.meta.hot.dispose(dispose);
}
