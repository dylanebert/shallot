import { Orbit, type Plugin } from "@dylanebert/shallot";
import { installHarness } from "@dylanebert/shallot/harness";
import { CAPTURE, SUN_FACING } from "./conditions";

const CapturePlugin: Plugin = {
    name: "Capture",
    initialize(state) {
        const harness = installHarness(state);
        harness.run = async () => {
            const query = new URLSearchParams(location.search);
            const condition = query.get("condition");
            if (condition === SUN_FACING.name) {
                const camera = state.only([Orbit]);
                Orbit.yaw.set(camera, SUN_FACING.camera.yaw);
                Orbit.pitch.set(camera, SUN_FACING.camera.pitch);
            }
            const requestedTime = query.get("time");
            const parsedTime = requestedTime === null ? Number.NaN : Number(requestedTime);
            const captureTime =
                Number.isFinite(parsedTime) && parsedTime >= 0 ? parsedTime : CAPTURE.time;
            while (state.time.elapsed < captureTime) {
                await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
            }
            state.pause();
            const canvas = document.querySelector("canvas");
            const claim = query.get("claim");
            const deviceChecks = claim
                ? await (
                      {
                          fold: () => import("./verification/ocean-fold"),
                          shading: () => import("./verification/ocean-shading"),
                          slope: () => import("./verification/ocean-slope"),
                          foam: () => import("./verification/ocean-foam"),
                          solar: () => import("./verification/ocean-solar"),
                      } as const
                  )
                      [claim as "fold" | "shading" | "slope" | "foam" | "solar"]?.()
                      .then((module) => module.runDeviceClaim(state))
                : [];
            const checks = [
                {
                    name: "matched 1280x720 capture viewport",
                    ok: innerWidth === CAPTURE.width && innerHeight === CAPTURE.height,
                    detail: `${innerWidth}x${innerHeight}`,
                },
                {
                    name: "matched ocean time reached",
                    ok: state.time.elapsed >= captureTime,
                    detail: `${state.time.elapsed} >= ${captureTime}`,
                },
                {
                    name: "named capture condition is known",
                    ok: condition === null || condition === SUN_FACING.name,
                    detail: condition ?? "default",
                },
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
