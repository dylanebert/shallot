import type { Plugin } from "@dylanebert/shallot";
import { installHarness } from "@dylanebert/shallot/harness";
import { CAPTURE } from "./conditions";

const CapturePlugin: Plugin = {
    name: "Capture",
    initialize(state) {
        const harness = installHarness(state);
        harness.run = async () => {
            while (state.time.elapsed < CAPTURE.time) {
                await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
            }
            state.pause();
            const canvas = document.querySelector("canvas");
            const checks = [
                {
                    name: "matched 1280x720 capture viewport",
                    ok: innerWidth === CAPTURE.width && innerHeight === CAPTURE.height,
                    detail: `${innerWidth}x${innerHeight}`,
                },
                { name: "matched ocean time reached", ok: state.time.elapsed >= CAPTURE.time },
                { name: "ocean canvas rendered", ok: Boolean(canvas?.width && canvas.height) },
            ];
            return { ok: checks.every((check) => check.ok), checks };
        };
    },
};

export default CapturePlugin;
