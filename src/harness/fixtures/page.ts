// The capture fixture's page: it installs the published harness protocol and makes its own stepped
// assertion in the page, consuming the public capture contract exactly as a project does. The driver
// transports the resulting verdict; it never asserts for this page.
//
// The serve command's `--fail` makes this same page fail on purpose, so the driver's surfacing of a page error, the serve
// command's output, the GPU diagnostics slot, the failing sub-check data and the retained artifact has a
// witness that does not need a broken product.

import type { Check, Verdict } from "@dylanebert/shallot/harness";
import { CAPTURE_CONTRACT, captureFrame } from "@dylanebert/shallot/harness/capture";
import { type PixelProbe, pixelProbePass, probePixels } from "@dylanebert/shallot/harness/pixels";
import { buildScene, SCENE_TAG, SCENE_TAG_PIXELS } from "./scene";

const TAG_PROBE: PixelProbe = {
    name: "tag reaches the final canvas",
    // The drawn region is derived from the fixture's own NDC quad and the declared capture geometry, so a
    // blank frame at the right size cannot pass: it needs most of the region actually filled.
    minPixels: Math.floor(SCENE_TAG_PIXELS * 0.9),
    minSpan: Math.floor(CAPTURE_CONTRACT.height / 2) - 1,
    r: [SCENE_TAG[0] - 8, SCENE_TAG[0] + 8],
    g: [SCENE_TAG[1] - 8, SCENE_TAG[1] + 8],
    b: [SCENE_TAG[2] - 8, SCENE_TAG[2] + 8],
};

async function main(): Promise<void> {
    const failing = (window as { __fixtureFail?: boolean }).__fixtureFail === true;
    const scene = await buildScene();
    await scene.step();
    const target: Window["__harness"] = {
        ready: true,
        async run(): Promise<Verdict> {
            const checks: Check[] = [];
            const first = await captureFrame(scene.canvas);
            const second = await captureFrame(scene.canvas);
            const identity = `${first.width}x${first.height}`;
            checks.push({
                name: "capture is the declared geometry",
                ok:
                    first.width === CAPTURE_CONTRACT.width &&
                    first.height === CAPTURE_CONTRACT.height,
                detail: `captured ${identity}, declared ${CAPTURE_CONTRACT.width}x${CAPTURE_CONTRACT.height}`,
                data: { width: first.width, height: first.height },
            });
            const shots = [first, second].map((shot) =>
                probePixels(shot.rgba, shot.width, shot.height, TAG_PROBE),
            );
            checks.push({
                name: "tag reaches the final canvas",
                ok: !failing && shots.every((shot) => pixelProbePass(shot, TAG_PROBE)),
                detail: `${shots.map((shot) => `${shot.pixels} px over ${shot.width}x${shot.height}`).join("; ")}, need ${TAG_PROBE.minPixels} px and ${TAG_PROBE.minSpan} span${failing ? " (deliberate failure)" : ""}`,
                data: { first: shots[0].pixels, second: shots[1].pixels },
            });
            checks.push({
                name: "repeated captures of one state agree",
                ok:
                    shots[0].pixels === shots[1].pixels &&
                    shots[0].width === shots[1].width &&
                    shots[0].height === shots[1].height,
                detail: `${shots[0].pixels} then ${shots[1].pixels} px`,
                data: { first: shots[0].pixels, second: shots[1].pixels },
            });
            return {
                ok: checks.every((entry) => entry.ok),
                checks,
                tick: scene.tick(),
                captureIdentity: identity,
            };
        },
    };
    window.__harness = target;
    if (failing) {
        // A real page error and a real console error, so the driver's surfacing is witnessed rather than
        // asserted about.
        console.error("fixture console error: deliberate failure requested");
        setTimeout(() => {
            throw new Error("fixture page error: deliberate failure requested");
        }, 0);
    }
}

main().catch((error) => {
    console.error(`fixture failed to boot: ${error instanceof Error ? error.message : error}`);
});
