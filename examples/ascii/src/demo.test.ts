import { resolve } from "node:path";
import { build, Compute, GlazePlugin, PartPlugin, probeBuffer, Time } from "@dylanebert/shallot";
import { CellsPlugin, cellsGridFor, OrbitPlugin } from "@dylanebert/shallot/extras";
import { check } from "@dylanebert/shallot/harness/check";
import { Camera, computeViewProj, type View, Views } from "@dylanebert/shallot/rendering";
import { SearPlugin } from "@dylanebert/shallot/standard/rendering";

const SCENE = resolve(import.meta.dir, "../public/scenes/ascii.scene");
const WIDTH = 256;
const HEIGHT = 256;

function project(eid: number, point: readonly [number, number, number]): [number, number] {
    const matrix = new Float32Array(16);
    computeViewProj(eid, WIDTH / HEIGHT, matrix);
    const [x, y, z] = point;
    const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
    return [((clipX / clipW) * 0.5 + 0.5) * WIDTH, (0.5 - (clipY / clipW) * 0.5) * HEIGHT];
}

function nearbyGlyphs(
    words: Uint32Array,
    cols: number,
    rows: number,
    point: readonly [number, number],
): number[] {
    const cx = Math.floor((point[0] / WIDTH) * cols);
    const cy = Math.floor((point[1] / HEIGHT) * rows);
    const glyphs: number[] = [];
    for (let y = Math.max(0, cy - 1); y <= Math.min(rows - 1, cy + 1); y++) {
        for (let x = Math.max(0, cx - 1); x <= Math.min(cols - 1, cx + 1); x++) {
            glyphs.push(words[(y * cols + x) * 3]);
        }
    }
    return glyphs;
}

function mode(values: readonly number[]): { glyph: number; count: number } {
    const counts = new Map<number, number>();
    for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
    const [glyph, count] = [...counts].sort((a, b) => b[1] - a[1])[0];
    return { glyph, count };
}

check(
    "ascii cube faces select three glyph regions",
    {
        claim: "ascii's actual scene selects distinct Cells glyphs on its cube's three visible faces",
        size: "integration",
        requires: ["gpu"],
        subject: "examples/ascii/public/scenes/ascii.scene",
    },
    async () => {
        const peerModule = "bun-webgpu";
        const peer = (await import(peerModule)) as { setupGlobals(): Promise<void> };
        await peer.setupGlobals();

        const app = await build({
            defaults: false,
            plugins: [OrbitPlugin, PartPlugin, SearPlugin, GlazePlugin, CellsPlugin],
            scene: SCENE,
        });
        let present: GPUTexture | undefined;
        const device = Compute.device;
        if (!device) {
            app.dispose();
            throw new Error("ascii has no GPU device");
        }
        const camera = app.state.only([Camera]);
        try {
            present = device.createTexture({
                label: "ascii-test-present",
                size: [WIDTH, HEIGHT],
                format: "bgra8unorm",
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
            });
            Views.set(camera, {
                canvas: null,
                context: { getCurrentTexture: () => present! } as unknown as GPUCanvasContext,
                width: WIDTH,
                height: HEIGHT,
                clientWidth: WIDTH,
                clientHeight: HEIGHT,
                viewportIndex: 0,
                framebuffer: null,
                present: null,
                depth: null,
                tag: null,
                slot: 0,
                observer: null,
                stamp: app.state.stamp(camera),
            } satisfies View);

            app.state.step(Time.FIXED_DT);
            app.state.step(Time.FIXED_DT);
            const grid = cellsGridFor(camera);
            if (!grid) throw new Error("Cells did not produce a grid for the scene camera");
            const raw = Compute.root.unwrap(grid.buffer);
            const probe = await probeBuffer(device, raw);
            const words = new Uint32Array(probe.bytes);
            const centers = [
                project(camera, [0, 0, 0.5]),
                project(camera, [0.5, 0, 0]),
                project(camera, [0, 0.5, 0]),
            ];
            const regions = centers.map((point) =>
                mode(nearbyGlyphs(words, grid.cols, grid.rows, point)),
            );
            if (new Set(regions.map(({ glyph }) => glyph)).size !== 3) {
                throw new Error(
                    `cube faces share a glyph: ${regions.map(({ glyph }) => glyph).join(", ")}`,
                );
            }
            if (regions.some(({ count }) => count < 4)) {
                throw new Error(
                    `cube face glyph regions are too small: ${regions.map(({ count }) => count).join(", ")}`,
                );
            }
        } finally {
            present?.destroy();
            app.dispose();
        }
    },
);
