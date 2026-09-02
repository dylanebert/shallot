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
            const claim = new URLSearchParams(location.search).get("claim");
            const deviceChecks = claim
                ? await (
                      {
                          fold: () => import("./verification/ocean-fold"),
                          shading: () => import("./verification/ocean-shading"),
                          slope: () => import("./verification/ocean-slope"),
                      } as const
                  )
                      [claim as "fold" | "shading" | "slope"]?.()
                      .then((module) => module.runDeviceClaim(state))
                : [];
            const checks = [
                {
                    name: "matched 1280x720 capture viewport",
                    ok: innerWidth === CAPTURE.width && innerHeight === CAPTURE.height,
                    detail: `${innerWidth}x${innerHeight}`,
                },
                { name: "matched ocean time reached", ok: state.time.elapsed >= CAPTURE.time },
                { name: "ocean canvas rendered", ok: Boolean(canvas?.width && canvas.height) },
                ...(
                    deviceChecks ?? [{ name: `known ocean device claim: ${claim}`, pass: false }]
                ).map((check) => ({ ...check, ok: check.pass })),
            ];
            return { ok: checks.every((check) => check.ok), checks };
        };
    },
};

export default CapturePlugin;
