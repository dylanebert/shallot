import { expect } from "bun:test";
import { CAPTURE_CONTRACT } from "@dylanebert/shallot/harness/capture";
import { check } from "@dylanebert/shallot/harness/check";
import {
    HOST_LAUNCHES,
    LAUNCH_MODES,
    type LaunchSeat,
    launchOptions,
    launchPlan,
} from "@dylanebert/shallot/harness/launch";
import {
    type AdapterFacts,
    adapterIdentity,
    classifyAdapter,
    resolveSeat,
    SEATS,
} from "@dylanebert/shallot/harness/seat";

// The identities are the ones the two real observers report: Chromium's masked info, Chromium's
// developer-features info, Chromium's forced fallback, and the Bun peer's device.
const CHROMIUM_MASKED: AdapterFacts = {
    present: true,
    info: { vendor: "apple", architecture: "metal-3", device: "", description: "" },
};
const CHROMIUM_REAL: AdapterFacts = {
    present: true,
    info: {
        vendor: "apple",
        architecture: "metal-3",
        device: "0x0000",
        description: "Apple M4 Max",
        isFallbackAdapter: false,
    },
};
const CHROMIUM_FALLBACK: AdapterFacts = {
    present: true,
    info: {
        vendor: "google",
        architecture: "swiftshader",
        device: "0xc0de",
        description: "SwiftShader Device (LLVM 10.0.0)",
        isFallbackAdapter: true,
    },
};
const BUN_REAL: AdapterFacts = {
    present: true,
    info: {
        vendor: "apple",
        architecture: "metal-3",
        device: "apple-m4-max",
        description: "Metal driver on macOS",
        isFallbackAdapter: false,
    },
};

function plan(host: string, seat: LaunchSeat = "chromium") {
    const resolved = launchPlan(host, seat);
    if ("refused" in resolved) throw new Error(resolved.refused);
    return resolved;
}

check(
    "adapter classification separates absent, fallback, unidentified and real adapters",
    {
        claim: "the seat resolver labels a software or unidentifiable adapter as a real device, so a SwiftShader run would report a real-adapter result",
        subject: "src/harness/seat.ts",
    },
    () => {
        expect(classifyAdapter({ present: false }).class).toBe("absent");
        expect(classifyAdapter({ present: true }).class).toBe("unidentified");
        expect(classifyAdapter({ present: true, info: {} }).class).toBe("unidentified");
        expect(classifyAdapter({ present: true, info: { vendor: "  " } }).class).toBe(
            "unidentified",
        );
        expect(classifyAdapter(CHROMIUM_FALLBACK).class).toBe("fallback");
        expect(classifyAdapter(CHROMIUM_REAL).class).toBe("real");
        expect(classifyAdapter(CHROMIUM_MASKED).class).toBe("real");
        expect(classifyAdapter(BUN_REAL).class).toBe("real");
        // Each non-real case refuses with its own reason, and a real one carries none.
        const reasons = [
            classifyAdapter({ present: false }).reason,
            classifyAdapter({ present: true }).reason,
            classifyAdapter(CHROMIUM_FALLBACK).reason,
        ];
        expect(new Set(reasons).size).toBe(3);
        expect(reasons.every((reason) => typeof reason === "string" && reason.length > 0)).toBe(
            true,
        );
        expect(classifyAdapter(CHROMIUM_REAL).reason).toBeUndefined();
        // The identity travels with the verdict and names the device, not the word "gpu".
        expect(adapterIdentity(CHROMIUM_REAL.info)).toBe("apple metal-3 0x0000 Apple M4 Max");
        expect(adapterIdentity(undefined)).toBe("unidentified");
    },
);

check(
    "a software adapter that does not set the fallback flag still refuses",
    {
        claim: "a software adapter whose fallback flag reads false or absent passes as a real device on identity alone",
        subject: "src/harness/seat.ts",
    },
    () => {
        const named = (description: string): AdapterFacts => ({
            present: true,
            info: { vendor: "google", description, isFallbackAdapter: false },
        });
        for (const description of [
            "SwiftShader Device (LLVM 10.0.0)",
            "llvmpipe (LLVM 15.0.0, 256 bits)",
            "lavapipe",
            "Microsoft Basic Render Driver",
            "Software Rasterizer",
        ]) {
            expect(classifyAdapter(named(description)).class).toBe("fallback");
        }
        // Exclusivity: a real device name is not swept up by the marker list.
        for (const description of ["Apple M4 Max", "NVIDIA GeForce RTX 4090", "AMD Radeon Pro"]) {
            expect(classifyAdapter(named(description)).class).toBe("real");
        }
    },
);

check(
    "one seat never substitutes for another",
    {
        claim: "a real Bun device satisfies the chromium seat, a browser satisfies the gpu seat, a headless browser or a bare declaration satisfies display, or a headed browser satisfies chromium, so one seat's evidence would be reported as another's",
        subject: "src/harness/seat.ts",
    },
    () => {
        const browser = {
            launch: plan("darwin"),
            adapter: CHROMIUM_REAL,
            capture: { identity: CAPTURE_CONTRACT },
        };
        const headed = { launch: plan("linux", "display"), adapter: CHROMIUM_REAL };
        // The truth table: each seat is satisfied by its own facts and by nothing else.
        const facts = {
            cpu: {},
            gpu: { device: BUN_REAL },
            chromium: { browser },
            display: { display: { source: "declared display", browser: headed } },
        } as const;
        for (const seat of SEATS) {
            for (const [name, given] of Object.entries(facts)) {
                const resolved = resolveSeat(seat, given, CAPTURE_CONTRACT);
                // `cpu` is the absence of a requirement, so every fact set satisfies it.
                expect(resolved.ok).toBe(seat === "cpu" || seat === name);
            }
        }
        expect(resolveSeat("gpu", { browser }, CAPTURE_CONTRACT)).toEqual({
            ok: false,
            reason: "gpu seat unavailable: no in-process WebGPU device was probed",
        });
        expect(resolveSeat("display", { browser }, CAPTURE_CONTRACT)).toEqual({
            ok: false,
            reason: "display seat unavailable: no headed display is declared",
        });
        // A headed browser never grants chromium, even on a real adapter at the capture contract.
        const headedChromium = resolveSeat(
            "chromium",
            { browser: { ...headed, capture: { identity: CAPTURE_CONTRACT } } },
            CAPTURE_CONTRACT,
        );
        expect(headedChromium).toEqual({
            ok: false,
            reason: "chromium seat unavailable: a headed launch never grants the chromium seat, which runs headless",
        });
        // Display needs both the declaration and a headed launch that reaches a real adapter: headed alone,
        // the declaration alone, a headless launch on the declaration, or a fallback adapter each refuse.
        const displayRefusals = [
            resolveSeat("display", { browser: headed }),
            resolveSeat("display", { display: { source: "declared display" } }),
            resolveSeat("display", {
                display: { source: "declared display", browser: { ...browser } },
            }),
            resolveSeat("display", {
                display: { source: "declared display", browser: { launch: headed.launch } },
            }),
            resolveSeat("display", {
                display: {
                    source: "declared display",
                    browser: { launch: headed.launch, adapter: CHROMIUM_FALLBACK },
                },
            }),
        ];
        expect(displayRefusals.every((resolved) => !resolved.ok)).toBe(true);
        const displayReasons = displayRefusals.map((resolved) =>
            resolved.ok ? "" : resolved.reason,
        );
        // Headed-alone and declaration-alone differ; the headless and declaration-alone cases share the
        // missing-headed-launch reason.
        expect(new Set(displayReasons).size).toBe(4);
        expect(displayReasons[4]).toContain("fallback adapter");
    },
);

check(
    "a missing or fallback premise refuses on every seat that needs it",
    {
        claim: "a missing adapter, fallback adapter, missing launch path or off-contract capture resolves a seat instead of refusing it",
        subject: "src/harness/seat.ts",
    },
    () => {
        const launch = plan("darwin");
        const refusals = [
            resolveSeat("gpu", { device: { present: false } }),
            resolveSeat("gpu", { device: CHROMIUM_FALLBACK }),
            resolveSeat("chromium", { browser: { adapter: CHROMIUM_REAL } }, CAPTURE_CONTRACT),
            resolveSeat("chromium", { browser: { launch } }, CAPTURE_CONTRACT),
            resolveSeat(
                "chromium",
                { browser: { launch, adapter: { present: false } } },
                CAPTURE_CONTRACT,
            ),
            resolveSeat(
                "chromium",
                { browser: { launch, adapter: CHROMIUM_FALLBACK } },
                CAPTURE_CONTRACT,
            ),
            resolveSeat(
                "chromium",
                { browser: { launch, adapter: CHROMIUM_REAL } },
                CAPTURE_CONTRACT,
            ),
            resolveSeat(
                "chromium",
                {
                    browser: {
                        launch,
                        adapter: CHROMIUM_REAL,
                        capture: { identity: { ...CAPTURE_CONTRACT, height: 719 } },
                    },
                },
                CAPTURE_CONTRACT,
            ),
        ];
        expect(refusals.every((resolved) => !resolved.ok)).toBe(true);
        // Distinct premises produce distinct refusals, so a verdict names which one was missing.
        const reasons = refusals.map((resolved) => (resolved.ok ? "" : resolved.reason));
        expect(new Set(reasons).size).toBe(refusals.length);
        expect(reasons.some((reason) => reason.includes("fallback adapter"))).toBe(true);
        expect(reasons.some((reason) => reason.includes("1280x719"))).toBe(true);
        // Non-vacuity: the same facts at the declared contract resolve.
        expect(
            resolveSeat(
                "chromium",
                {
                    browser: {
                        launch,
                        adapter: CHROMIUM_REAL,
                        capture: { identity: CAPTURE_CONTRACT },
                    },
                },
                CAPTURE_CONTRACT,
            ).ok,
        ).toBe(true);
    },
);

check(
    "a host launch declaration cannot change the launch mode or a seat's meaning",
    {
        claim: "a host launch declaration can select a headed launch or make a fallback adapter acceptable, so a host could pass a seat it does not have",
        subject: ["src/harness/launch.ts", "src/harness/launch.json"],
    },
    () => {
        const hosts = Object.keys(HOST_LAUNCHES).sort();
        // Pin the declared population against an empty scan, and against a host nobody declared.
        expect(hosts).toEqual(["darwin", "linux", "win32"]);
        expect(launchPlan("freebsd")).toEqual({
            refused:
                "no declared headless Chromium launch path for host freebsd; declared hosts are darwin, linux, win32",
        });
        expect(LAUNCH_MODES).toEqual({ chromium: "headless", display: "headed" });
        for (const host of hosts) {
            const resolved = plan(host);
            expect(resolved.mode).toBe("headless");
            expect(resolved.channel).toBe("chromium");
            expect(launchOptions(resolved).headless).toBe(true);
            const display = plan(host, "display");
            expect(display.mode).toBe("headed");
            expect(launchOptions(display).headless).toBe(false);
            expect(display.args).toEqual(resolved.args);
            // A plan's own fields cannot select headed for chromium: options read the seat's policy.
            expect(
                launchOptions({ ...resolved, mode: "headed", adapterEvidence: "proven" }).headless,
            ).toBe(true);
            // No declaration carries a mode, a headless field or a capability of its own.
            expect(Object.keys(HOST_LAUNCHES[host]).sort()).toEqual(["adapterEvidence", "note"]);
            expect(Object.keys(HOST_LAUNCHES[host].adapterEvidence).sort()).toEqual([
                "headed",
                "headless",
            ]);
            // A host's proven headed evidence never grants chromium on a headed plan.
            expect(
                resolveSeat(
                    "chromium",
                    {
                        browser: {
                            launch: display,
                            adapter: CHROMIUM_REAL,
                            capture: { identity: CAPTURE_CONTRACT },
                        },
                    },
                    CAPTURE_CONTRACT,
                ).ok,
            ).toBe(false);
            // The declaration's own evidence never grants the seat: the fallback adapter refuses under
            // every host, including the one where a real adapter is proven.
            expect(
                resolveSeat(
                    "chromium",
                    {
                        browser: {
                            launch: resolved,
                            adapter: CHROMIUM_FALLBACK,
                            capture: { identity: CAPTURE_CONTRACT },
                        },
                    },
                    CAPTURE_CONTRACT,
                ).ok,
            ).toBe(false);
            // And a declared-but-unproven host still resolves on an observed real adapter, because the
            // observation is what counts.
            expect(
                resolveSeat(
                    "chromium",
                    {
                        browser: {
                            launch: resolved,
                            adapter: CHROMIUM_REAL,
                            capture: { identity: CAPTURE_CONTRACT },
                        },
                    },
                    CAPTURE_CONTRACT,
                ).ok,
            ).toBe(true);
        }
        // Headless evidence is proven only on the authoring seat; headed evidence only where a headed launch
        // reached a real adapter. Neither grants a seat; they record what was observed.
        expect(HOST_LAUNCHES.darwin.adapterEvidence).toEqual({
            headless: "proven",
            headed: "unproven",
        });
        expect(HOST_LAUNCHES.linux.adapterEvidence).toEqual({
            headless: "unproven",
            headed: "proven",
        });
        expect(HOST_LAUNCHES.win32.adapterEvidence).toEqual({
            headless: "unproven",
            headed: "unproven",
        });
        expect(plan("linux").adapterEvidence).toBe("unproven");
        expect(plan("linux", "display").adapterEvidence).toBe("proven");
    },
);
