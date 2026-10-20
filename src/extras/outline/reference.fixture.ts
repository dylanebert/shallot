import { beforeAll } from "bun:test";
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

export const size = 32;
export const widths = [2.5, 4, 5.5, 9.5];
export type Pixel = readonly [number, number];
export const masks: { name: string; pixels: Pixel[] }[] = [
    { name: "thin line", pixels: Array.from({ length: 18 }, (_, i) => [15, i + 7]) },
    {
        name: "three-seed meeting",
        pixels: [
            [20, 10],
            [20, 18],
            [14, 20],
        ],
    },
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
export const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] },
]);
export let render: (
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
    const texture = () => {
        const t = device.createTexture({
            size: [size, size],
            format: "rgba16float",
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC,
        });
        world.own(t);
        return t;
    };
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
    const attr = texture(),
        scene = texture(),
        output = texture();
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
        const attrs = new Uint16Array(size * size * 4);
        for (const [x, y] of pixels) {
            const at = y * size + x;
            attrs[at * 4] = 0x3c00;
            attrs[at * 4 + 3] = new Uint16Array(new Float16Array([width]).buffer)[0]!;
        }
        const offset = MAX_JFA_PASSES - jfaPassCount(width);
        device.queue.writeTexture(
            { texture: seed[offset % 2]! },
            initialSeeds(pixels),
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

function smoothstep(low: number, high: number, value: number): number {
    const t = Math.max(0, Math.min(1, (value - low) / (high - low)));
    return t * t * (3 - 2 * t);
}
export function coverage(distance: number, width: number): number {
    return smoothstep(0, 1, distance) * (1 - smoothstep(width - 1, width, distance));
}
export function initialSeeds(pixels: Pixel[]): Uint16Array<ArrayBuffer> {
    const seeds = new Uint16Array(size * size * 2).fill(30000);
    for (const [x, y] of pixels) seeds.set([x, y], (y * size + x) * 2);
    return seeds;
}
export function squaredDistance(seeds: Uint16Array, x: number, y: number): number {
    const at = (y * size + x) * 2;
    return (x - seeds[at]!) ** 2 + (y - seeds[at + 1]!) ** 2;
}

/** CPU nine-neighbour flood, including its non-nearest results.
 * The independent ladder has Golus's ceil(log2(width + 1)) levels, ending at 1.
 * The current seed wins ties; otherwise the first in y-then-x order wins.
 */
export function jumpFlood(pixels: Pixel[], width: number): Uint16Array<ArrayBuffer> {
    let seeds = initialSeeds(pixels);
    const first = 2 ** (Math.ceil(Math.log2(Math.max(1, Math.min(64, width)) + 1)) - 1);
    for (let step = first; step >= 1; step /= 2) {
        const next = new Uint16Array(seeds.length);
        for (let y = 0; y < size; y++)
            for (let x = 0; x < size; x++) {
                let best = (y * size + x) * 2;
                let bestDistance = squaredDistance(seeds, x, y);
                for (let oy = -1; oy <= 1; oy++)
                    for (let ox = -1; ox <= 1; ox++) {
                        const qx = x + ox * step,
                            qy = y + oy * step;
                        if (qx < 0 || qy < 0 || qx >= size || qy >= size) continue;
                        const at = (qy * size + qx) * 2;
                        const distance = (x - seeds[at]!) ** 2 + (y - seeds[at + 1]!) ** 2;
                        if (distance < bestDistance) {
                            best = at;
                            bestDistance = distance;
                        }
                    }
                next.set(seeds.subarray(best, best + 2), (y * size + x) * 2);
            }
        seeds = next;
    }
    return seeds;
}
