import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { type CostRow, costCells, costReport } from "./cost-report";

const template = readFileSync(new URL("./cost.html", import.meta.url), "utf8");
const row: CostRow = {
    scene: "<scene>",
    quality: "Medium",
    adapter: {
        vendor: "apple",
        architecture: "metal-3",
        device: "0x0000",
        description: "Apple M4 Max",
    },
    browser: "</pre><script>untrusted</script>",
    resolution: "3024×1964",
    timestampQuery: "available",
    samples: 1000,
    frameMs: 4.3,
    spanSamples: { "render/standard:ssao-evaluate/elapsed_gpu": 999 },
    spansMs: { "render/standard:ssao-evaluate/elapsed_gpu": 2.345678 },
};

test("saved cost JSON renders the live page's cells without running its measurement script", () => {
    const html = costReport(template, [row]);
    expect(html).not.toContain("<script");
    expect(html).toContain("Complete — saved measurements");
    expect(html).toContain("&lt;scene&gt;");
    expect(html).toContain("4.300");
    expect(html).toContain("render/standard:ssao-evaluate/elapsed_gpu: 2.345678 ms");
    expect(html).toContain("&lt;/pre&gt;&lt;script&gt;untrusted&lt;/script&gt;");
    expect(html.match(/<tr>/g)?.length).toBe(2);
});

test("missing timestamp-query is reported as unavailable, not as zero GPU cost", () => {
    const cells = costCells({
        ...row,
        timestampQuery: "unavailable: frame time alone",
        spansMs: {},
    });
    expect(cells[3]).toBe("4.300");
    expect(cells[4]).toBe("unavailable: frame time alone");
    expect(costCells({ ...row, quality: "Off", spansMs: {} })[4]).toBe("—");
});
