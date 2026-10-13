import { expect, test } from "bun:test";
import { oklchFromLinear, PALETTES, srgbChannel } from "./palette";
import { reportHtml } from "./report";

test("current-in-range repairs only overflowing albedo and ink, preserving hue and alpha", () => {
    const baseline = PALETTES.current;
    const repaired = PALETTES["current-in-range"];
    for (const key of Object.keys(baseline.materials) as (keyof typeof baseline.materials)[]) {
        const before = baseline.materials[key];
        const after = repaired.materials[key];
        if (key === "blue" || key === "orange") {
            const [l, c, h] = oklchFromLinear(before);
            const [ll, cc, hh] = oklchFromLinear(after);
            expect(ll).toBeLessThan(l);
            expect(cc).toBeCloseTo(c, 6);
            expect(hh).toBeCloseTo(h, 4);
            expect(after[3]).toBe(before[3]);
            expect(Math.max(...after.slice(0, 3).map(srgbChannel))).toBeCloseTo(240, 8);
        } else expect(after).toEqual(before);
    }
    expect(repaired.clearColor).toBe(baseline.clearColor);
    expect(repaired.ui.panel).toBe(baseline.ui.panel);
    expect(repaired.ui.border).toBe(baseline.ui.border);
    expect(repaired.ui.ink).toBe("#f1f1f1");
    expect(repaired.textColor).toBe(0xf1f1f1);
});

test("dev-orange-neutral keeps accents but gives large surfaces current's mid-neutral lightness", () => {
    const palette = PALETTES["dev-orange-neutral"];
    const orange = PALETTES["dev-orange"];
    for (const key of ["blue", "orange", "greyboxLift", "embedded"] as const)
        expect(palette.materials[key]).toEqual(orange.materials[key]);
    for (const key of [
        "blueGround",
        "greyboxGround",
        "greyboxStep",
        "greyboxTower",
        "greyboxPerch",
    ] as const) {
        const [l, c] = oklchFromLinear(palette.materials[key]);
        expect(l).toBeCloseTo(oklchFromLinear(PALETTES.current.materials[key])[0], 6);
        expect(c).toBeLessThanOrEqual(0.0120001);
    }
    expect(palette.materials.trim).toEqual(orange.materials.greyboxTower);
    expect(palette.ui).toEqual(orange.ui);
});

test("all five candidates meet hard bounds while the report retains theme-reference violations", () => {
    for (const [name, palette] of Object.entries(PALETTES)) {
        if (name === "current") continue;
        for (const value of Object.values(palette.materials)) {
            for (const channel of value.slice(0, 3)) {
                expect(srgbChannel(channel)).toBeGreaterThanOrEqual(50);
                expect(srgbChannel(channel)).toBeLessThanOrEqual(240);
            }
        }
    }
    const html = reportHtml();
    expect(html.match(/<img /g)).toHaveLength(24);
    expect(html).toContain("Theme reference");
    expect(html).toContain("outside");
});
