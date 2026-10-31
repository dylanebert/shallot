import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { homedir, hostname, release } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import sharp from "sharp";
import { environmentPin, viewerPin } from "./pin";
import { modelPath, scenes } from "../../examples/gltf/src/scenes";

const commit = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
const directory = join(homedir(), "gltf-look", commit.slice(0, 8));
const categories = ["core", "issues", "testing", "written"];
const names = Object.keys(scenes) as (keyof typeof scenes)[];
const ordered = categories
    .flatMap((category) => names.filter((name) => scenes[name].tags[0] === category))
    .filter((name) => !process.env.GLTF_LOOK_SCENE || name === process.env.GLTF_LOOK_SCENE);
if (ordered.length === 0) throw new Error("GLTF_LOOK_SCENE must name a scene in the catalog");

test("every scene beside the pinned Sample Viewer at its bounds-fitted camera", async ({
    browser,
}) => {
    test.setTimeout(0);
    await mkdir(directory, { recursive: true });
    const records = [];
    for (const scene of ordered) {
        const reference = await browser.newPage({ viewport: { width: 960, height: 720 } });
        await reference.goto(
            `http://127.0.0.1:4176/?model=${encodeURIComponent(`/models/${modelPath(scene)}`)}`,
        );
        reference.on("pageerror", (error) => {
            throw error;
        });
        await reference.waitForFunction(() => "capture" in window, null, {
            timeout: TIMEOUT.browser,
        });
        const camera = await reference.evaluate(
            () =>
                (
                    window as unknown as {
                        capture: {
                            renderer: string;
                            position: number[];
                            fov: number;
                            near: number;
                            far: number;
                        };
                    }
                ).capture,
        );
        expect(camera.renderer).toMatch(/Apple/);
        expect(camera.renderer).not.toMatch(/SwiftShader|llvmpipe|software/i);
        const right = await reference.locator("canvas").screenshot();
        const page = await browser.newPage({ viewport: { width: 960, height: 720 } });
        await page.addInitScript(() => {
            const request = navigator.gpu.requestAdapter.bind(navigator.gpu);
            navigator.gpu.requestAdapter = async (options) => {
                const adapter = await request(options);
                (window as unknown as { adapter: unknown }).adapter = adapter && {
                    vendor: adapter.info.vendor,
                    architecture: adapter.info.architecture,
                    device: adapter.info.device,
                    description: adapter.info.description,
                    isFallbackAdapter: adapter.info.isFallbackAdapter,
                };
                return adapter;
            };
        });
        await page.goto(
            `/tests/look.html?scene=${scene}&camera=${encodeURIComponent(JSON.stringify(camera))}`,
        );
        page.on("pageerror", (error) => {
            throw error;
        });
        await page.waitForFunction(() => "captureReady" in window, null, {
            timeout: TIMEOUT.browser,
        });
        await page.waitForFunction(
            () =>
                performance
                    .getEntriesByType("measure")
                    .some((entry) => entry.name.endsWith(":first-update")),
            null,
            { timeout: TIMEOUT.browser },
        );
        const adapter = await page.evaluate(
            () => (window as unknown as { adapter: GPUAdapterInfo }).adapter,
        );
        expect(adapter.isFallbackAdapter).toBe(false);
        expect(JSON.stringify(adapter)).toMatch(/apple/i);
        const left = await page.locator("canvas").screenshot();
        await sharp({ create: { width: 1920, height: 720, channels: 4, background: "black" } })
            .composite([
                { input: left, left: 0, top: 0 },
                { input: right, left: 960, top: 0 },
            ])
            .png()
            .toFile(join(directory, `${scene}.png`));
        records.push({ scene, camera, adapter });
        await page.close();
        await reference.close();
    }
    const metadata = {
        host: `${hostname()} Darwin ${release()}`,
        commit,
        sampleViewer: viewerPin,
        environment: `Footprint Court (environment submodule ${environmentPin}); linear tonemapping, exposure 1, IBL intensity 1`,
        differences:
            "Shallot uses its warm sun and cool ambient, not image-based lighting; its ground and truck placement are omitted for the same raw model and camera. Sample Viewer uses its default environment and tonemapping; Shallot retains its own. No visual equivalence is asserted.",
        records,
    };
    await writeFile(join(directory, "run.json"), JSON.stringify(metadata, null, 2));
    await writeFile(
        join(directory, "index.html"),
        `<!doctype html><html lang="en"><meta charset="utf-8"><title>glTF look ${commit.slice(0, 8)}</title><style>body{background:#191816;color:#eee;font:16px sans-serif;margin:24px}img{width:100%;height:auto}section{margin-bottom:32px}</style><h1>Shallot · Khronos Sample Viewer v1.0.10</h1><p>Left: Shallot. Right: Sample Viewer. Each side 960 × 720.</p><p>${metadata.host}<br>Shallot ${commit}<br>Sample Viewer ${viewerPin}<br>${metadata.environment}</p><p>${metadata.differences}</p>${categories
            .map(
                (category) =>
                    `<h2>${category}</h2>${ordered
                        .filter((scene) => scenes[scene].tags[0] === category)
                        .map(
                            (scene) =>
                                `<section><h3>${scene}</h3><img src="${scene}.png" alt="${scene}: Shallot left, Sample Viewer right"></section>`,
                        )
                        .join("")}`,
            )
            .join("")}</html>`,
    );
    console.log(`Capture pairs: ${directory}`);
});
