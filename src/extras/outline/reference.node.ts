import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Meshes, MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, CameraMode } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { probeTexture } from "../../engine/runtime";
import { StandardRenderer } from "../../standard/rendering";
import { Outline } from ".";
import {
    coverage,
    initialSeeds,
    jumpFlood,
    masks,
    render,
    size,
    squaredDistance,
    subjects,
    widths,
} from "./reference.fixture";

setDefaultTimeout(TIMEOUT.node);

for (const mask of masks)
    for (const width of widths) {
        test(`${mask.name}, width ${width}: exact CPU jump flood and coverage within one display code`, async () => {
            const actual = await render(mask.pixels, width);
            const expected = jumpFlood(mask.pixels, width);
            expect(actual.nearest).toEqual(expected);
            const values = new Float16Array(actual.coverage.buffer);
            let maxCoverageError = 0;
            for (let y = 0; y < size; y++)
                for (let x = 0; x < size; x++) {
                    const distance = Math.sqrt(squaredDistance(expected, x, y));
                    maxCoverageError = Math.max(
                        maxCoverageError,
                        Math.abs(values[(y * size + x) * 4 + 3]! - coverage(distance, width)),
                    );
                }
            console.info(JSON.stringify({ mask: mask.name, width, maxCoverageError }));
            // Alpha is linear coverage, not sRGB.
            expect(maxCoverageError).toBeLessThanOrEqual(1 / 255);
            if (mask.name === "three-seed meeting" && width === 5.5) {
                // The jump flood is not an exact distance transform: preserve this known error.
                expect(squaredDistance(actual.nearest, 16, 15)).toBe(29);
                expect(
                    Math.min(...mask.pixels.map(([x, y]) => (16 - x) ** 2 + (15 - y) ** 2)),
                ).toBe(25);
                expect(coverage(Math.sqrt(29), width)).toBeCloseTo(0.037, 3);
                expect(coverage(5, width)).toBe(0.5);
            }
        });
    }

for (const mask of masks) {
    test(`${mask.name}: the real orthographic mesh mask seeds exactly the covered pixels before flooding`, async () => {
        const { world } = subjects()[0];
        const { device } = world.gpu;
        const snapshot = device.createTexture({
            size: [size, size],
            format: "rg16uint",
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
        });
        world.own(snapshot);
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera, { mode: CameraMode.Orthographic, size: size / 2, antialias: 0 });
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: size, height: size });
        // One world unit is one pixel. Quad edges fall on pixel boundaries.
        const vertices: number[] = [],
            indices: number[] = [];
        for (const [x, y] of mask.pixels) {
            const base = vertices.length / 8;
            for (const [dx, dy] of [
                [0, 1],
                [1, 1],
                [1, 0],
                [0, 0],
            ])
                vertices.push(x + dx! - size / 2, size / 2 - y - dy!, 0, 0, 0, 0, 1, 0);
            indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
        }
        const meshes = world.resource(Meshes);
        const mesh = meshes.add({
            name: mask.name,
            vertices: new Float32Array(vertices),
            indices: new Uint32Array(indices),
        });
        const entity = world.create();
        world.add(entity, Transform);
        world.add(entity, MeshInstance, { mesh });
        world.add(entity, Outline, { width: 5.5 });
        const views = new WeakMap<GPUTextureView, GPUTexture>();
        const createTexture = device.createTexture.bind(device);
        const createEncoder = device.createCommandEncoder.bind(device);
        let finalSeed: GPUTexture | undefined;
        let copies = 0;
        device.createTexture = (descriptor) => {
            if (!descriptor.label?.startsWith("outline-seed")) return createTexture(descriptor);
            const texture = createTexture({
                ...descriptor,
                usage: descriptor.usage | GPUTextureUsage.COPY_SRC,
            });
            // The production seven-slot ladder always finishes in seed B.
            if (descriptor.label === `outline-seedB-${camera}`) finalSeed = texture;
            const createView = texture.createView.bind(texture);
            texture.createView = (descriptor) => {
                const view = createView(descriptor);
                views.set(view, texture);
                return view;
            };
            return texture;
        };
        device.createCommandEncoder = (descriptor) => {
            const encoder = createEncoder(descriptor);
            const begin = encoder.beginRenderPass.bind(encoder);
            encoder.beginRenderPass = (descriptor) => {
                const pass = begin(descriptor);
                if (descriptor.label !== `outline-mask/${camera}`) return pass;
                const source = views.get(
                    Array.from(descriptor.colorAttachments)[0]!.view as GPUTextureView,
                )!;
                const end = pass.end.bind(pass);
                pass.end = () => {
                    end();
                    encoder.copyTextureToTexture({ texture: source }, { texture: snapshot }, [
                        size,
                        size,
                    ]);
                    copies++;
                };
                return pass;
            };
            return encoder;
        };
        try {
            world.step(0);
            expect(copies).toBe(1);
            const actual = new Uint16Array(
                (await probeTexture(world, snapshot)).bytes as ArrayBuffer,
            );
            expect(actual).toEqual(initialSeeds(mask.pixels));
            const flooded = new Uint16Array(
                (await probeTexture(world, finalSeed!)).bytes as ArrayBuffer,
            );
            expect(flooded).toEqual(jumpFlood(mask.pixels, 5.5));
        } finally {
            device.createTexture = createTexture;
            device.createCommandEncoder = createEncoder;
            world.destroy(entity);
            world.destroy(camera);
            meshes.remove(mesh);
            world.step(0);
        }
    });
}
