import { expect } from "bun:test";
import { check } from "@dylanebert/shallot/harness/check";
import type { AdapterFacts } from "../engine/runtime/adapter";
import floor from "./browser.json" with { type: "json" };
import { launchPlan } from "./launch";
import { resolveSeat, type SeatResolution } from "./seat";

check(
    "the launch plan contains operational data only",
    {
        claim: "configured OS launch targets retain the Chromium channel and browser arguments without historical adapter evidence",
    },
    () => {
        for (const host of ["darwin", "linux", "win32"]) {
            const plan = launchPlan(host);
            expect(plan).toEqual({
                host,
                seat: "display",
                channel: "chromium",
                args: floor.args,
            });
            // This is the boundary regression: adapter support belongs to the observed run, not launch data.
            expect("adapterEvidence" in plan).toBe(false);
        }

        const refused = launchPlan("freebsd");
        expect(refused).toEqual({
            refused:
                "no headed Chromium launch configuration for platform freebsd; configured platforms are darwin, linux, win32",
        });
    },
);

check(
    "supplied adapter facts decide the display seat",
    {
        claim: "a display seat refuses absent, unidentified and fallback adapters and accepts only a supplied positively identified real adapter",
    },
    () => {
        const plan = launchPlan("darwin");
        if ("refused" in plan) throw new Error(plan.refused);
        const resolve = (adapter: AdapterFacts | undefined): SeatResolution =>
            resolveSeat("display", {
                display: {
                    source: "DP-1",
                    browser: { launch: plan, adapter },
                },
            });

        const absent = resolve({ present: false });
        expect(absent.ok).toBe(false);
        if (!absent.ok) expect(absent.reason).toContain("no WebGPU adapter");

        const unidentified = resolve({ present: true, info: {} });
        expect(unidentified.ok).toBe(false);
        if (!unidentified.ok) expect(unidentified.reason).toContain("no identity");

        const fallback = resolve({
            present: true,
            info: { vendor: "Google", device: "SwiftShader Device" },
        });
        expect(fallback.ok).toBe(false);
        if (!fallback.ok) expect(fallback.reason).toContain("fallback adapter");

        expect(resolve({ present: true, info: { vendor: "Apple", device: "M2" } })).toEqual({
            ok: true,
            detail: "headed chromium on real adapter Apple M2 via DP-1",
        });

        const noBrowser = resolveSeat("display", {
            display: { source: "DP-1" },
        });
        expect(noBrowser.ok).toBe(false);
        if (!noBrowser.ok) expect(noBrowser.reason).toContain("no headed Chromium launch");
    },
);
