import { expect, test } from "@playwright/test";

// Drive the voxel mesher's device gate: load the app, wait for it to warm and expose `window.__voxelGate`,
// run it on the real GPU, and assert every check passes (and the page raised no error). The checks
// themselves live in `src/gate.ts` against the published surface — this driver is the only part Playwright
// touches. One session, phases within one test (the Playwright structure rule).

interface Check {
    name: string;
    pass: boolean;
    detail: string;
}

test("voxel mesher gate — watertight, deterministic, carve (real GPU)", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    // standalone runtime — the gate only touches GPU buffers, no editor/scene writes.
    await page.goto("/");
    // the boot plugin generates the terrain then installs the hook once the mesher buffers exist (see
    // boot.ts). A cold vite build is slower than the warm path, so allow longer than steady-state needs.
    await page.waitForFunction(() => typeof window.__voxelGate === "function", null, {
        timeout: 120_000,
    });

    const checks = (await page.evaluate(() =>
        (window as unknown as { __voxelGate: () => Promise<Check[]> }).__voxelGate(),
    )) as Check[];

    expect(errors, errors.join("\n")).toEqual([]);
    expect(checks.length).toBeGreaterThan(0);
    for (const c of checks) {
        expect(c.pass, `${c.name}: ${c.detail}`).toBe(true);
    }
});
