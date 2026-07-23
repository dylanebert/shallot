import { run } from "@dylanebert/shallot";
import { installHarness } from "@dylanebert/shallot/harness";

// The red-proof fixture. A truly blank app: the default camera + renderer clear to background, no
// geometry, no UI. Its harness reports ok:true, so the only thing that can fail the run is verify's
// pixel gate reading the blank canvas as rendered:false. scripts/flows.ts asserts exactly that FAILURE
// (an expected-fail) — the standing proof that the gate goes red on a canvas that drew nothing, and the
// runnable oracle behind the pixel-honest `rendered` verdict. It deliberately does NOT declare noRender:
// that is the opt-out this proves the gate still catches a real blank without.
const scene = `<scene>
    <a ambient-light="intensity: 0.6" />
    <a camera sear transform />
</scene>`;

const { state, dispose } = await run({ plugins: [], scene });
const harness = installHarness(state);
harness.run = async () => ({ ok: true, checks: [{ name: "booted", ok: true }] });

// HMR re-runs this module — dispose the old State + RAF loop, or each edit stacks another.
if (import.meta.hot) {
    import.meta.hot.dispose(() => dispose());
}
