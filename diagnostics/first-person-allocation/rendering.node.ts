import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { CEILING } from "../../scripts/test-tiers";
import { sampleAllocation, windowBytes } from "./allocation";

setDefaultTimeout(CEILING.node);

test("steady Sprite and Text update paths allocate no sampled JavaScript heap after warm-up", async () => {
    const sample = await sampleAllocation(resolve(import.meta.dir, "rendering.entry.ts"), {
        warm: 6000,
        frames: 120,
    });
    console.info(
        `[rendering-allocation] ${sample.runtime} ${JSON.stringify(
            sample.windows.map((window) => ({
                label: window.label,
                bytes: windowBytes(window),
                sites: window.sites,
                spriteTextSites: window.sites.filter((row) =>
                    /src\/extras\/(?:sprite|text)\//.test(row.site),
                ),
                optimizations: window.optimizations,
            })),
        )}`,
    );
    expect(sample.control.length).toBeGreaterThan(0);
    expect(windowBytes({ sites: sample.control })).toBeGreaterThan(0);
    expect(sample.windows.map((window) => window.label)).toEqual([
        "after warm 6000",
        "after warm 12000",
        "A/A repeat",
    ]);
    const subjectOptimizations = sample.windows.flatMap((window) =>
        (window.optimizations ?? []).filter((name) => name !== "setImmediate"),
    );
    expect(subjectOptimizations).toEqual([]);
    expect(
        sample.windows.map((window) =>
            window.sites.filter((row) => /src\/extras\/(?:sprite|text)\//.test(row.site)),
        ),
    ).toEqual([[], [], []]);
});
