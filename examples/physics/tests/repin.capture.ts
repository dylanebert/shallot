import { expect, test } from "playwright/test";
import { capture } from "./capture";

for (const width of [1280, 390])
    for (const scene of [
        "hullCulling",
        "edgeAxisWedge",
        "encroachCast",
        "hull2D",
        "spinningCapsule",
        "safetyFactor",
        "sleep",
        "spinner",
        "benchmarkMeshDrop",
        "benchmarkVillage",
        "bounceImpulse",
        "boxRestitution",
        "restitutionPropagation",
        "rotatedBoxRestitution",
        "singleBoxRestitution",
        "singleSphereRestitution",
        "sphereRestitution",
        "sphereStackRestitution",
        "twoBoxRestitution",
    ])
        test(`${scene} re-pin look at ${width}`, async ({ browser }, info) => {
            const context = await browser.newContext({
                viewport: { width, height: width === 390 ? 844 : 720 },
                hasTouch: width === 390,
            });
            try {
                const page = await context.newPage();
                await page.goto(`${info.project.use.baseURL}/?scene=${scene}`);
                await expect(
                    page.getByRole("button", { name: "Scene", exact: true }),
                ).toBeEnabled();
                if (["hull2D", "encroachCast", "hullCulling", "edgeAxisWedge"].includes(scene))
                    await page.evaluate(
                        () =>
                            new Promise<void>((resolve) => {
                                let frames = 20;
                                const frame = () => {
                                    if (--frames) requestAnimationFrame(frame);
                                    else resolve();
                                };
                                requestAnimationFrame(frame);
                            }),
                    );
                await capture(page, `${scene}-${width}`);
            } finally {
                await context.close();
            }
        });
