import type { Page } from "@playwright/test";

// Shared real-GPU adapter probe: every voxel Playwright spec (the correctness gate, the structural perf
// gate, the measurement instrument) needs the same "skip on a software adapter" check before waiting on
// the app's boot hook — duplicated across three specs, this is the one source. Not itself a `*.spec.ts`,
// so Playwright's own testMatch never picks it up as a test file.

// Software rasterizers by the name they report in `GPUAdapterInfo`: Chromium's SwiftShader, Mesa's two,
// and D3D's WARP. Deliberately a name list rather than a capability probe — SwiftShader clears shallot's
// whole base floor (all three `BASE_FEATURES` plus 10 storage buffers per stage, measured under WSL
// 2026-08-10), so nothing about the floor distinguishes it, and it is only the *execution* that dies.
// Bias narrow: an unlisted software adapter re-crashes loudly, while an over-broad pattern would skip this
// gate on real hardware and report green having tested nothing.
export const SOFTWARE = /swiftshader|llvmpipe|lavapipe|warp|basic render/i;

/** the adapter's self-reported identity, or `""` when the browser hands out none. */
export const adapterName = (page: Page): Promise<string> =>
    page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "";
        const { vendor, architecture, device, description } = adapter.info;
        return [vendor, architecture, device, description].filter(Boolean).join(" ");
    });
