import { expect } from "bun:test";
import { CAPTURE_CONTRACT } from "@dylanebert/shallot/harness/capture";
import { check } from "@dylanebert/shallot/harness/check";
import {
    HIDDEN_WINDOW_CLASS,
    type LaunchSeat,
    launchMode,
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
        claim: "a real Bun device satisfies the chromium seat, a browser satisfies the gpu seat, a headless browser or a bare declaration satisfies display, or a display browser satisfies chromium, so one seat's evidence would be reported as another's",
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
        // A display launch never grants chromium, even on a real adapter at the capture contract.
        const headedChromium = resolveSeat(
            "chromium",
            { browser: { ...headed, capture: { identity: CAPTURE_CONTRACT } } },
            CAPTURE_CONTRACT,
        );
        expect(headedChromium).toEqual({
            ok: false,
            reason: "chromium seat unavailable: a display launch never grants the chromium seat",
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
        expect(launchPlan("freebsd")).toHaveProperty("refused");
        expect(launchMode("darwin", "chromium")).toBe("headless");
        expect(launchMode("darwin", "display")).toBe("headed");
        const resolved = plan("darwin");
        expect(resolved.channel).toBe("chromium");
        expect(launchOptions(resolved).headless).toBe(true);
        const display = plan("darwin", "display");
        expect(launchOptions(display).headless).toBe(false);
        expect(display.args).toEqual(resolved.args);
        // A plan carries no mode of its own, so seat resolution and launch options both derive it.
        expect("mode" in resolved || "mode" in display).toBe(false);
        expect(launchOptions({ ...resolved, adapterEvidence: "proven" }).headless).toBe(true);
        const chromium = (launch: typeof resolved, adapter: AdapterFacts) =>
            resolveSeat(
                "chromium",
                { browser: { launch, adapter, capture: { identity: CAPTURE_CONTRACT } } },
                CAPTURE_CONTRACT,
            ).ok;
        // Declared evidence never grants the seat: a display plan and a fallback adapter both refuse,
        // and a real adapter on the headless plan resolves because the observation is what counts.
        expect(chromium(display, CHROMIUM_REAL)).toBe(false);
        expect(chromium(resolved, CHROMIUM_FALLBACK)).toBe(false);
        expect(chromium(resolved, CHROMIUM_REAL)).toBe(true);
    },
);

check(
    "a headed chromium launch grants the seat only hidden by the compositor",
    {
        claim: "a host whose evidence is headed launches chromium headless, or grants the chromium seat to a headed window no compositor rule hid, so a run either reaches only SwiftShader or takes the person's desktop",
        subject: ["src/harness/launch.ts", "src/harness/launch.json", "src/harness/seat.ts"],
    },
    () => {
        // The mode comes from the host's evidence: Linux proves only headed, so chromium launches headed
        // there, with the hidden class; macOS proves headless and launches with no class.
        expect(launchMode("linux", "chromium")).toBe("headed");
        const headed = plan("linux");
        const options = launchOptions(headed);
        expect(options.headless).toBe(false);
        expect(options.args).toContain(`--class=${HIDDEN_WINDOW_CLASS}`);
        expect(launchOptions(plan("darwin")).args).not.toContain(`--class=${HIDDEN_WINDOW_CLASS}`);
        expect(launchOptions(plan("linux", "display")).args).not.toContain(
            `--class=${HIDDEN_WINDOW_CLASS}`,
        );
        const hidden = {
            address: "0x1",
            class: HIDDEN_WINDOW_CLASS,
            workspace: "special:gate",
            own: true,
        };
        const chromium = (facts: object) =>
            resolveSeat(
                "chromium",
                {
                    browser: {
                        launch: headed,
                        adapter: CHROMIUM_REAL,
                        capture: { identity: CAPTURE_CONTRACT },
                        ...facts,
                    },
                },
                CAPTURE_CONTRACT,
            );
        expect(chromium({ hidden: { windows: [hidden], activeClass: "Alacritty" } }).ok).toBe(true);
        // Each way no rule took refuses by name: no read-back, no window of the class, one shown, or focus.
        expect(chromium({})).toMatchObject({
            ok: false,
            reason: expect.stringContaining("read-back"),
        });
        expect(chromium({ hidden: { windows: [], activeClass: "" } })).toMatchObject({
            ok: false,
            reason: expect.stringContaining(`no window of class ${HIDDEN_WINDOW_CLASS}`),
        });
        // Another run's hidden window of the class is not this launch's: with none of its own, it refuses.
        expect(
            chromium({ hidden: { windows: [{ ...hidden, own: false }], activeClass: "" } }),
        ).toMatchObject({
            ok: false,
            reason: expect.stringContaining("opened by this launch"),
        });
        expect(
            chromium({
                hidden: {
                    windows: [hidden, { ...hidden, address: "0x2", workspace: "3" }],
                    activeClass: "",
                },
            }),
        ).toMatchObject({ ok: false, reason: expect.stringContaining("0x2 on 3") });
        expect(
            chromium({ hidden: { windows: [hidden], activeClass: HIDDEN_WINDOW_CLASS } }),
        ).toMatchObject({ ok: false, reason: expect.stringContaining("active window") });
    },
);
