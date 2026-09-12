import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { installAllocProbe, metricOf } from "./verify";

const GC_METRICS = ["MinorGCCount", "MajorGCCount", "MinorGCDuration", "MajorGCDuration"] as const;
const G4_FIXTURES =
    process.env.S5B_G4_FIXTURES ??
    "/tmp/shallot-cut-the-gate-s5b-witnesses/g4-allocation-fixtures.json";

type Metrics = { metrics: { name: string; value: number }[] };
type Cdp = {
    send(method: string, params?: unknown): Promise<unknown>;
    detach(): Promise<void>;
};
type Binding = (windowMs: number) => Promise<unknown>;

function missingMetrics(): Metrics {
    return { metrics: [{ name: "JSHeapUsedSize", value: 1_000_000 }] };
}

function fakePage(metrics: Metrics): { page: unknown; binding: () => Binding } {
    let exposed: Binding | undefined;
    const cdp: Cdp = {
        async send(method) {
            if (method === "Performance.getMetrics") return metrics;
            if (method === "HeapProfiler.stopSampling")
                return {
                    profile: {
                        head: {
                            callFrame: {
                                functionName: "fixture",
                                url: "fixture.ts",
                                lineNumber: 0,
                            },
                            selfSize: 0,
                        },
                    },
                };
            return {};
        },
        async detach() {},
    };
    const page = {
        context: () => ({ newCDPSession: async () => cdp }),
        exposeFunction: async (_name: string, fn: Binding) => {
            exposed = fn;
        },
    };
    return {
        page,
        binding: () => {
            if (!exposed) throw new Error("production allocation binding was not exposed");
            return exposed;
        },
    };
}

describe("G5 real-response membership witness", () => {
    test("all eight retained G4 endpoint pairs omit every GC metric while the old reader returns zero", () => {
        const fixture = JSON.parse(readFileSync(G4_FIXTURES, "utf8")) as {
            runs: { allocationReceipts: { m0: Metrics; m1: Metrics }[] }[];
        };
        const receipts = fixture.runs.flatMap((run) => run.allocationReceipts);
        expect(receipts).toHaveLength(8);
        for (const receipt of receipts) {
            for (const endpoint of [receipt.m0, receipt.m1]) {
                expect(endpoint.metrics.some((metric) => metric.name === "JSHeapUsedSize")).toBe(
                    true,
                );
                for (const name of GC_METRICS) expect(metricOf(endpoint, name)).toBe(0);
            }
        }
    });
});

describe("G5 production allocation call-shape refusal witness", () => {
    test("missing GC membership refuses instead of returning the original zero wall result", async () => {
        const { page, binding } = fakePage(missingMetrics());
        await installAllocProbe(page as never);
        await expect(binding()(0)).rejects.toThrow("GC observation unavailable");
    });
});
