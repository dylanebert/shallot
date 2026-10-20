import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { probeTexture } from "../../engine/runtime";
import { DEFAULT_PLUGINS } from "../../standard";
import { OutlinePlugin } from ".";
import {
    compositeKernel,
    compositeLayout,
    fullscreenVs,
    jfaFs,
    jfaLayout,
    jfaPassCount,
    jfaStep,
    MAX_JFA_PASSES,
    WORKGROUP,
} from "./passes";

setDefaultTimeout(TIMEOUT.node);
const size = 32;
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] },
]);
type Pixel = readonly [number, number];
const masks: { name: string; pixels: Pixel[] }[] = [
    { name: "thin line", pixels: Array.from({ length: 18 }, (_, i) => [15, i + 7]) },
    {
        name: "concave corner",
        pixels: Array.from(
            { length: size * size },
            (_, i) => [i % size, Math.floor(i / size)] as const,
        ).filter(([x, y]) => x >= 8 && x < 24 && y >= 8 && y < 24 && (x < 12 || y >= 20)),
    },
    {
        name: "nearby silhouettes",
        pixels: [
            ...Array.from({ length: 12 }, (_, i) => [12, i + 9] as const),
            ...Array.from({ length: 12 }, (_, i) => [17, i + 11] as const),
        ],
    },
];
let render: (
    pixels: Pixel[],
    width: number,
) => Promise<{
    coverage: Uint16Array;
    nearest: Uint16Array;
}>;
beforeAll(() => {
    const { world } = subjects()[0];
    const { device, root } = world.gpu;
    const info = device.adapterInfo;
    console.info(
        JSON.stringify({
            adapter: {
                vendor: info.vendor,
                architecture: info.architecture,
                device: info.device,
                description: info.description,
                isFallbackAdapter: info.isFallbackAdapter,
            },
        }),
    );
    const texture = (format: GPUTextureFormat) => {
        const t = device.createTexture({
            size: [size, size],
            format,
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC,
        });
        world.own(t);
        return t;
    };
    // rg16uint cannot be a storage attachment.
    const seed = [0, 1].map(() => {
        const t = device.createTexture({
            size: [size, size],
            format: "rg16uint",
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC,
        });
        world.own(t);
        return t;
    });
    const attr = texture("rgba16float");
    const scene = texture("rgba16float");
    const output = texture("rgba16float");
    device.queue.writeTexture(
        { texture: scene },
        new Uint16Array(size * size * 4),
        { bytesPerRow: size * 8 },
        [size, size],
    );
    const jfa = root.unwrap(
        root.createRenderPipeline({
            vertex: fullscreenVs,
            fragment: jfaFs,
            targets: { format: "rg16uint" },
            primitive: { topology: "triangle-list", cullMode: "none" },
        }),
    );
    const composite = root.unwrap(root.createComputePipeline({ compute: compositeKernel }));
    const groups = Array.from({ length: MAX_JFA_PASSES }, (_, k) =>
        root.unwrap(
            root.createBindGroup(jfaLayout, {
                seed: seed[k % 2]!.createView(),
                step: root.createBuffer(d.f32, jfaStep(k)).$usage("uniform"),
            }),
        ),
    );
    const compositeGroup = root.unwrap(
        root.createBindGroup(compositeLayout, {
            scene: scene.createView(),
            seed: seed[MAX_JFA_PASSES % 2]!.createView(),
            attr: attr.createView(),
            output: output.createView(),
        }),
    );
    render = async (pixels, width) => {
        const seeds = new Uint16Array(size * size * 2).fill(30000);
        const attrs = new Uint16Array(size * size * 4);
        for (const [x, y] of pixels) {
            const at = y * size + x;
            seeds[at * 2] = x;
            seeds[at * 2 + 1] = y;
            attrs[at * 4] = 0x3c00;
            // All fixed widths are exactly representable half floats.
            attrs[at * 4 + 3] = new Uint16Array(new Float16Array([width]).buffer)[0]!;
        }
        const offset = MAX_JFA_PASSES - jfaPassCount(width);
        device.queue.writeTexture(
            { texture: seed[offset % 2]! },
            seeds,
            { bytesPerRow: size * 4 },
            [size, size],
        );
        device.queue.writeTexture({ texture: attr }, attrs, { bytesPerRow: size * 8 }, [
            size,
            size,
        ]);
        const encoder = device.createCommandEncoder();
        for (let k = offset; k < MAX_JFA_PASSES; k++) {
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: seed[1 - (k % 2)]!.createView(),
                        loadOp: "clear",
                        storeOp: "store",
                        clearValue: { r: 30000, g: 30000, b: 0, a: 0 },
                    },
                ],
            });
            pass.setPipeline(jfa);
            pass.setBindGroup(0, groups[k]!);
            pass.draw(3);
            pass.end();
        }
        const pass = encoder.beginComputePass();
        pass.setPipeline(composite);
        pass.setBindGroup(0, compositeGroup);
        pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP), Math.ceil(size / WORKGROUP));
        pass.end();
        device.queue.submit([encoder.finish()]);
        return {
            coverage: new Uint16Array((await probeTexture(world, output)).bytes as ArrayBuffer),
            nearest: new Uint16Array(
                (await probeTexture(world, seed[MAX_JFA_PASSES % 2]!)).bytes as ArrayBuffer,
            ),
        };
    };
});

// Rong and Tan, "Jump Flooding in GPU with Applications to Voronoi Diagram and
// Distance Transform" (2006), describes errors where seed regions meet. Nearby
// silhouettes and the concave corner exercise those meetings; no pixels are exempt.
function smoothstep(low: number, high: number, value: number): number {
    const t = Math.max(0, Math.min(1, (value - low) / (high - low)));
    return t * t * (3 - 2 * t);
}

// Coverage is alpha, not sRGB: one display code is 1/255 of linear coverage.
const halfBits = new Uint16Array(1);
const halfValue = new Float16Array(halfBits.buffer);
for (const mask of masks)
    for (const width of [2.5, 4, 5.5, 9.5]) {
        test(`${mask.name}, width ${width}: exact band geometry and coverage within one display code`, async () => {
            const actual = await render(mask.pixels, width);
            const mismatches: {
                pixel: Pixel;
                squaredDistance: number;
                floodSquaredDistance: number;
                expected: number;
                actual: number;
            }[] = [];
            let maxCoverageError = 0;
            for (let y = 0; y < size; y++)
                for (let x = 0; x < size; x++) {
                    // maskFragment seeds integer pixel coordinates, not subpixel edge estimates.
                    // Exhaustive reference, independent of the jump ladder and chosen seed.
                    const squaredDistance = Math.min(
                        ...mask.pixels.map(([sx, sy]) => (x - sx) ** 2 + (y - sy) ** 2),
                    );
                    const distance = Math.sqrt(squaredDistance);
                    // Shallot's composite, not a binary dilation or Golus's different edge fade.
                    const expected =
                        smoothstep(0, 1, distance) * (1 - smoothstep(width - 1, width, distance));
                    const at = y * size + x;
                    const floodSquaredDistance =
                        (x - actual.nearest[at * 2]!) ** 2 + (y - actual.nearest[at * 2 + 1]!) ** 2;
                    halfBits[0] = actual.coverage[at * 4 + 3]!;
                    const coverage = halfValue[0]!;
                    const error = Math.abs(coverage - expected);
                    maxCoverageError = Math.max(maxCoverageError, error);
                    // The ladder only promises distances in band range. Even where the fade
                    // saturates, those distances must be exact; coverage tolerance hides no JFA error.
                    const wrongGeometry =
                        Math.min(squaredDistance, floodSquaredDistance) < width ** 2 &&
                        squaredDistance !== floodSquaredDistance;
                    if (wrongGeometry || coverage > 0 !== expected > 0 || error > 1 / 255)
                        mismatches.push({
                            pixel: [x, y],
                            squaredDistance,
                            floodSquaredDistance,
                            expected,
                            actual: coverage,
                        });
                }
            console.info(JSON.stringify({ mask: mask.name, width, maxCoverageError, mismatches }));
            expect(mismatches).toEqual([]);
        });
    }
