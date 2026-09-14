import { expect } from "bun:test";
import { resolve } from "node:path";
import { runBrowserCheck } from "@dylanebert/shallot/harness";
import { CAPTURE_CONTRACT } from "@dylanebert/shallot/harness/capture";
import { check } from "@dylanebert/shallot/harness/check";

// The two browser rows of the capture and seat contract, on the available macOS seat. The page owns its
// stepped assertion; these rows own the seat classification, the capture geometry and the surfacing of a
// failure. The fixture is engine-free on purpose: a render defect must not read as a capture defect.

const ROOT = resolve(import.meta.dir, "../..");
const SERVE = resolve(import.meta.dir, "fixtures/serve.ts");
const serveCommand = (extra: string[]) => (port: number) => [
    process.execPath,
    SERVE,
    "--port",
    String(port),
    ...extra,
];

check(
    "a headless browser row runs on a positively identified real adapter at the fixed capture identity",
    {
        claim: "the browser seat runs on an unidentified or fallback adapter, at an implicit viewport, or disagrees with itself across repeated captures of one stepped state",
        size: "integration",
        requires: ["chromium"],
        subject: [
            "src/harness/driver.ts",
            "src/harness/capture.ts",
            "src/harness/seat.ts",
            "src/harness/launch.ts",
        ],
    },
    async () => {
        const verdict = await runBrowserCheck(serveCommand([]));
        const failed = (verdict.checks ?? []).filter((entry) => !entry.ok);
        expect(failed.map((entry) => `${entry.name}: ${entry.detail}`)).toEqual([]);
        expect(verdict.ok).toBe(true);
        // The page's own three claims: declared geometry, tagged content, and repeated agreement.
        expect((verdict.checks ?? []).length).toBe(3);
        expect(verdict.captureIdentity).toBe(
            `${CAPTURE_CONTRACT.width}x${CAPTURE_CONTRACT.height}`,
        );
        // F6: the reproduction record names the seat this result actually came from.
        const record = verdict.reproduction;
        expect(record.launch).toBe("headless");
        expect(record.adapterClass).toBe("real");
        expect(record.adapter).not.toBe("unidentified");
        expect(record.adapter.toLowerCase()).not.toContain("swiftshader");
        expect(record.viewport).toBe(
            `${CAPTURE_CONTRACT.width}x${CAPTURE_CONTRACT.height}@${CAPTURE_CONTRACT.deviceScale}`,
        );
        expect(record.capture).toBe("final-canvas 1280x720@1 rgba8-tight");
        expect(record.chromium).toMatch(/^\d+\./);
        expect(record.runtime).toContain("chromium");
        expect(record.engine).toMatch(/^\d+\.\d+/);
        expect(record.host).toContain(process.platform);
        expect(record.tick).toBeGreaterThan(0);
        return verdict;
    },
);

check(
    "a failing browser row retains the page, server and sub-check evidence behind it",
    {
        claim: "a failing browser row reports only that it failed, discarding the page errors, serve-command output, failing sub-check data and the actual frame",
        size: "integration",
        requires: ["chromium"],
        subject: ["src/harness/driver.ts", "src/harness/verdict.ts"],
    },
    async () => {
        // A caller cannot ask for a headed browser at all: the option is refused, not silently ignored,
        // and the refusal happens before any browser launches.
        const headed = await runBrowserCheck(serveCommand([]), { headless: false } as never).then(
            () => null,
            (reason: Error) => reason.message,
        );
        expect(headed).toContain("`headless` is not a caller option");

        const error = await runBrowserCheck(serveCommand(["--fail"])).then(
            () => null,
            (reason: unknown) => reason,
        );
        expect(error).not.toBeNull();
        const failure = error as {
            message: string;
            reproduction: { adapterClass: string; capture: string };
            diagnostics: {
                pageErrors?: string[];
                serverLog?: string;
                artifacts?: string[];
                checks?: Array<{ name: string; detail?: string; data?: Record<string, number> }>;
            };
        };
        expect(failure.message).toContain("failing verdict");
        // The real page error and console error reached the verdict rather than being swallowed.
        const pageErrors = (failure.diagnostics.pageErrors ?? []).join(" | ");
        expect(pageErrors).toContain("fixture page error: deliberate failure requested");
        expect(pageErrors).toContain("fixture console error: deliberate failure requested");
        // The serve command's own output survives; it used to be discarded to /dev/null.
        expect(failure.diagnostics.serverLog ?? "").toContain("fixture serve listening on");
        // The failing sub-check keeps its structured data, and the frame itself is retained, bounded.
        const failedChecks = failure.diagnostics.checks ?? [];
        expect(failedChecks.map((entry) => entry.name)).toEqual(["tag reaches the final canvas"]);
        expect(failedChecks[0]?.data?.first).toBeGreaterThan(0);
        const artifacts = failure.diagnostics.artifacts ?? [];
        expect(artifacts.length).toBe(1);
        const artifact = Bun.file(artifacts[0] as string);
        expect(await artifact.exists()).toBe(true);
        expect(artifact.size).toBeGreaterThan(0);
        expect((artifacts[0] as string).startsWith(resolve(ROOT, ".artifacts"))).toBe(true);
        // The seat is still reported truthfully on a failure, not blanked.
        expect(failure.reproduction.adapterClass).toBe("real");
        expect(failure.reproduction.capture).toBe("final-canvas 1280x720@1 rgba8-tight");
        return { ok: true };
    },
);
