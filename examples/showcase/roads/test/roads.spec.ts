import { expect, type Page, test } from "@playwright/test";

// Drive the terrain generator's device gate: load the app, wait for it to warm and expose
// `window.__roadsGate`, run it on the real GPU, and assert every check passes (and the page raised no
// error). The checks themselves live in `src/gate.ts` against the published surface — this driver is the
// only part Playwright touches. One session, phases within one test (the Playwright structure rule).

interface Check {
    name: string;
    pass: boolean;
    detail: string;
}

// Software rasterizers by the name they report in `GPUAdapterInfo`: Chromium's SwiftShader, Mesa's two,
// and D3D's WARP. Deliberately a name list rather than a capability probe — SwiftShader clears shallot's
// whole base floor (all three `BASE_FEATURES` plus 10 storage buffers per stage, measured under WSL
// 2026-08-10), so nothing about the floor distinguishes it, and it is only the *execution* that dies.
// Bias narrow: an unlisted software adapter re-crashes loudly, while an over-broad pattern would skip this
// gate on real hardware and report green having tested nothing.
const SOFTWARE = /swiftshader|llvmpipe|lavapipe|warp|basic render/i;

/** the adapter's self-reported identity, or `""` when the browser hands out none. */
const adapterName = (page: Page): Promise<string> =>
    page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "";
        const { vendor, architecture, device, description } = adapter.info;
        return [vendor, architecture, device, description].filter(Boolean).join(" ");
    });

test("terrain generator gate — sized, deterministic, reseeds, not flat (real GPU)", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    // standalone runtime — the gate only touches GPU buffers, no editor/scene writes.
    await page.goto("/");

    // Full device testing: the generator needs a real GPU. Probe before waiting on the boot hook, because
    // on a software adapter the app never installs it and the wait burns its full 120 s before failing —
    // display-gated the same way `shallot verify`'s callers are, so a GPU-less host skips instead of reds.
    const adapter = await adapterName(page);
    console.log(`roads gate adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    // the boot plugin installs the hook once the terrain mesh is registered (see boot.ts). A cold vite
    // build is slower than the warm path, so allow longer than steady-state needs.
    await page.waitForFunction(() => typeof window.__roadsGate === "function", null, {
        timeout: 120_000,
    });

    const checks = (await page.evaluate(() =>
        (window as unknown as { __roadsGate: () => Promise<Check[]> }).__roadsGate(),
    )) as Check[];

    expect(errors, errors.join("\n")).toEqual([]);
    expect(checks.length).toBeGreaterThan(0);
    for (const c of checks) {
        expect(c.pass, `${c.name}: ${c.detail}`).toBe(true);
    }
});
