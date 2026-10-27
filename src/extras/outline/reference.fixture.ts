import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { fullscreenVertex } from "../../core/rendering";
import { precompile, type World } from "../../engine";
import { probeTexture } from "../../engine/runtime";
import { DEFAULT_PLUGINS } from "../../standard";
import { OutlinePlugin } from ".";
import {
    compositeHalfKernel,
    compositeKernel,
    compositeLayout,
    distanceFs,
    distanceLayout,
    jfaFs,
    jfaHalfFs,
    jfaLayout,
    jfaPassCount,
    jfaStep,
    MAX_JFA_PASSES,
    reduceFs,
    reduceLayout,
    WORKGROUP,
} from "./passes";

export const size = 32;
export const widths = [1.5, 2, 2.5, 3, 3.5, 4, 5.5, 7, 7.5, 8, 9.5];
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
export function outlineReference(path: string) {
    const subjects = gpuApps(path, [
        {
            defaults: false,
            plugins: [
                ...DEFAULT_PLUGINS,
                OutlinePlugin,
                { name: "OutlineReference", dependencies: [OutlinePlugin], initialize },
            ],
        },
    ]);
    let render: (
        pixels: Pixel[],
        width: number,
    ) => Promise<{
        coverage: Uint16Array;
        nearest: Uint16Array;
        distances?: Float32Array;
    }>;
    function initialize(world: World) {
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
        const halfSeed = [0, 1].map(() => {
            const t = device.createTexture({
                size: [size / 2, size / 2],
                format: "rg16uint",
                usage:
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.RENDER_ATTACHMENT |
                    GPUTextureUsage.COPY_SRC,
            });
            world.own(t);
            return t;
        });
        const distance = device.createTexture({
            size: [size / 2, size / 2],
            format: "r16float",
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.COPY_SRC,
        });
        world.own(distance);
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
            root
                .createRenderPipeline({
                    vertex: fullscreenVertex,
                    fragment: jfaFs,
                    targets: { format: "rg16uint" },
                    primitive: { topology: "triangle-list", cullMode: "none" },
                })
                .$name("outline-reference-jfa"),
        );
        const composite = root.unwrap(
            root
                .createComputePipeline({ compute: compositeKernel })
                .$name("outline-reference-composite"),
        );
        const renderPipeline = (fragment: typeof jfaFs, format: GPUTextureFormat) =>
            root.unwrap(
                root.createRenderPipeline({
                    vertex: fullscreenVertex,
                    fragment,
                    targets: { format },
                    primitive: { topology: "triangle-list", cullMode: "none" },
                }),
            );
        const halfJfa = renderPipeline(jfaHalfFs, "rg16uint");
        const reduce = renderPipeline(reduceFs, "rg16uint");
        const distancePipeline = root.unwrap(
            root.createRenderPipeline({
                vertex: fullscreenVertex,
                fragment: distanceFs,
                targets: { format: "r16float" },
                primitive: { topology: "triangle-list", cullMode: "none" },
            }),
        );
        const halfComposite = root.unwrap(
            root.createComputePipeline({ compute: compositeHalfKernel }),
        );
        precompile(world, "outline-reference", () => [
            jfa,
            composite,
            halfJfa,
            reduce,
            distancePipeline,
            halfComposite,
        ]);
        const halfGroups = Array.from({ length: MAX_JFA_PASSES }, (_, k) =>
            root.unwrap(
                root.createBindGroup(jfaLayout, {
                    seed: halfSeed[k % 2]!.createView(),
                    step: root.createBuffer(d.f32, jfaStep(k)).$usage("uniform"),
                }),
            ),
        );
        const reduceGroup = root.unwrap(
            root.createBindGroup(reduceLayout, { mask: seed[0]!.createView() }),
        );
        const distanceGroup = root.unwrap(
            root.createBindGroup(distanceLayout, {
                seed: halfSeed[MAX_JFA_PASSES % 2]!.createView(),
            }),
        );
        const halfCompositeGroup = root.unwrap(
            root.createBindGroup(compositeLayout, {
                distance: distance.createView(),
                scene: scene.createView(),
                seed: halfSeed[MAX_JFA_PASSES % 2]!.createView(),
                attr: attr.createView(),
                output: output.createView(),
            }),
        );
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
                distance: distance.createView(),
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
            const half = width >= 2;
            const fields = half ? halfSeed : seed;
            const offset = MAX_JFA_PASSES - jfaPassCount(half ? width / 2 : width);
            device.queue.writeTexture(
                { texture: seed[half ? 0 : offset % 2]! },
                initialSeeds(pixels),
                { bytesPerRow: size * 4 },
                [size, size],
            );
            device.queue.writeTexture({ texture: attr }, attrs, { bytesPerRow: size * 8 }, [
                size,
                size,
            ]);
            const encoder = device.createCommandEncoder();
            const fullscreen = (
                target: GPUTexture,
                pipeline: GPURenderPipeline,
                group: GPUBindGroup,
            ) => {
                const pass = encoder.beginRenderPass({
                    colorAttachments: [
                        { view: target.createView(), loadOp: "clear", storeOp: "store" },
                    ],
                });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, group);
                pass.draw(3);
                pass.end();
            };
            if (half) fullscreen(halfSeed[offset % 2]!, reduce, reduceGroup);
            for (let k = offset; k < MAX_JFA_PASSES; k++) {
                const pass = encoder.beginRenderPass({
                    colorAttachments: [
                        {
                            view: fields[1 - (k % 2)]!.createView(),
                            loadOp: "clear",
                            storeOp: "store",
                            clearValue: { r: 30000, g: 30000, b: 0, a: 0 },
                        },
                    ],
                });
                pass.setPipeline(half ? halfJfa : jfa);
                pass.setBindGroup(0, half ? halfGroups[k]! : groups[k]!);
                pass.draw(3);
                pass.end();
            }
            if (half) fullscreen(distance, distancePipeline, distanceGroup);
            const pass = encoder.beginComputePass();
            pass.setPipeline(half ? halfComposite : composite);
            pass.setBindGroup(0, half ? halfCompositeGroup : compositeGroup);
            pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP), Math.ceil(size / WORKGROUP));
            pass.end();
            device.queue.submit([encoder.finish()]);
            const field = half
                ? new Float16Array((await probeTexture(world, distance)).bytes as ArrayBuffer)
                : undefined;
            const distances = field
                ? Float32Array.from({ length: size * size }, (_, i) =>
                      attrs[i * 4 + 3]
                          ? 0
                          : sampleDistance(
                                field,
                                size / 2,
                                size / 2,
                                i % size,
                                Math.floor(i / size),
                            ),
                  )
                : undefined;
            return {
                distances,
                coverage: new Uint16Array((await probeTexture(world, output)).bytes as ArrayBuffer),
                nearest: new Uint16Array(
                    (await probeTexture(world, fields[MAX_JFA_PASSES % 2]!)).bytes as ArrayBuffer,
                ),
            };
        };
    }
    return { subjects, render: (pixels: Pixel[], width: number) => render(pixels, width) };
}

/** Bilinear clamp-to-edge sampling at full-resolution pixel centers of a 2× field. */
export function sampleDistance(
    field: ArrayLike<number>,
    w: number,
    h: number,
    x: number,
    y: number,
): number {
    const u = (x + 0.5) / 2 - 0.5,
        v = (y + 0.5) / 2 - 0.5;
    const ix = Math.floor(u),
        iy = Math.floor(v);
    const get = (x: number, y: number) =>
        field[Math.max(0, Math.min(h - 1, y)) * w + Math.max(0, Math.min(w - 1, x))]!;
    const a = u - ix,
        b = v - iy;
    return (
        (get(ix, iy) * (1 - a) + get(ix + 1, iy) * a) * (1 - b) +
        (get(ix, iy + 1) * (1 - a) + get(ix + 1, iy + 1) * a) * b
    );
}

/** Independent half-resolution seed reduction, bounded flood and half-float distance resolve. */
export function halfFlood(pixels: Pixel[], width: number) {
    const n = Math.ceil(size / 2);
    const mask = initialSeeds(pixels);
    let seeds = new Uint16Array(n * n * 2).fill(30000);
    for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++) {
            let best = Infinity;
            for (let dy = 0; dy < 2; dy++)
                for (let dx = 0; dx < 2; dx++) {
                    const at = ((2 * y + dy) * size + 2 * x + dx) * 2;
                    const dist =
                        (2 * x + 0.5 - mask[at]!) ** 2 + (2 * y + 0.5 - mask[at + 1]!) ** 2;
                    if (dist < best) {
                        best = dist;
                        seeds.set(mask.subarray(at, at + 2), (y * n + x) * 2);
                    }
                }
        }
    const first = 2 ** (Math.ceil(Math.log2(width / 2 + 1)) - 1);
    for (let step = first; step >= 1; step /= 2) {
        const next = new Uint16Array(seeds.length);
        for (let y = 0; y < n; y++)
            for (let x = 0; x < n; x++) {
                let best = (y * n + x) * 2;
                const distance = (at: number) =>
                    (2 * x + 0.5 - seeds[at]!) ** 2 + (2 * y + 0.5 - seeds[at + 1]!) ** 2;
                let bestD = distance(best);
                for (let dy = -1; dy <= 1; dy++)
                    for (let dx = -1; dx <= 1; dx++) {
                        const qx = x + dx * step,
                            qy = y + dy * step;
                        if (qx < 0 || qy < 0 || qx >= n || qy >= n) continue;
                        const at = (qy * n + qx) * 2,
                            dist = distance(at);
                        if (dist < bestD) {
                            best = at;
                            bestD = dist;
                        }
                    }
                next.set(seeds.subarray(best, best + 2), (y * n + x) * 2);
            }
        seeds = next;
    }
    const field = new Float16Array(n * n);
    for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++) {
            const at = (y * n + x) * 2;
            field[y * n + x] = Math.hypot(2 * x + 0.5 - seeds[at]!, 2 * y + 0.5 - seeds[at + 1]!);
        }
    const alpha = new Float32Array(size * size);
    for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++) {
            const at = (Math.floor(y / 2) * n + Math.floor(x / 2)) * 2;
            const valid = seeds[at]! < size && seeds[at + 1]! < size;
            alpha[y * size + x] =
                mask[(y * size + x) * 2] !== 30000 || !valid
                    ? 0
                    : coverage(sampleDistance(field, n, n, x, y), width);
        }
    return { seeds, alpha };
}

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
