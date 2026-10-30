import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, type MeshHandle, MeshPlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const cases = [
    // MetalRoughSpheres, Spheres.004/Primitive0: normalization must use the stored f32 extent.
    [0.03436123952269554, 0.1278132051229477, 0.03826630115509033],
    // Rounding the normalized input before packing selects 4104 instead of the nearest 4105.
    [-0.25, 1.75, -0.12473868578672409],
];
const handles: MeshHandle[] = [];
const apps = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            MeshPlugin,
            {
                name: "QuantizationMidpoint",
                dependencies: [MeshPlugin],
                initialize(world) {
                    for (const [i, values] of cases.entries())
                        handles.push(
                            world.resource(Meshes).add({
                                name: `midpoint-${i}`,
                                vertices: new Float32Array(
                                    values.flatMap((v) => [v, v, v, v, 0, 0, 1, v]),
                                ),
                                indices: new Uint32Array([0, 1, 2]),
                            }),
                        );
                },
            },
        ],
    },
]);
const ulp = (v: number) =>
    Math.abs(v) < 2 ** -126 ? 2 ** -149 : 2 ** (Math.floor(Math.log2(Math.abs(v))) - 23);

for (const [i, values] of cases.entries())
    test(`${i === 0 ? "MetalRoughSpheres" : "normalized midpoint"}: position and UV choose the nearest lattice point before f32 decode rounding`, async () => {
        const { world } = apps()[0];
        console.log("mesh quantization adapter", world.gpu.adapter);
        const mesh = world.resource(Meshes).get(handles[i])!;
        const vertices = await mesh.vertices.read();
        const base = (await mesh.indices.read())[mesh.indexBase];
        const quant = (await mesh.quant!.read())[vertices[base].y >>> 16];
        for (let vertex = 0; vertex < values.length; vertex++) {
            const packed = vertices[base + vertex];
            const codes = [
                packed.x & 65535,
                packed.x >>> 16,
                packed.y & 65535,
                packed.w & 65535,
                packed.w >>> 16,
            ];
            const offsets = [...quant.posOffset.xyz, quant.posOffset.w, quant.posScale.w];
            const extents = [...quant.posScale.xyz, ...quant.uvScale.xy];
            for (let axis = 0; axis < codes.length; axis++) {
                const offset = offsets[axis];
                const extent = extents[axis];
                const expected = values[vertex];
                expect(codes[axis]).toBe(
                    Math.round(Math.min(1, Math.max(0, (expected - offset) / extent)) * 65535),
                );
                const normalized = Math.fround(codes[axis] / 65535);
                const product = Math.fround(normalized * extent);
                const decoded = Math.fround(offset + product);
                const bound =
                    extent / 131070 +
                    (Math.abs(extent) * ulp(normalized)) / 2 +
                    ulp(product) / 2 +
                    ulp(decoded) / 2;
                expect(Math.abs(decoded - expected)).toBeLessThanOrEqual(bound);
            }
        }
    });
