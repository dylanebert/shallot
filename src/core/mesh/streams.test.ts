import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { pack2x16unorm } from "typegpu/std";
import { World } from "../../engine";
import { packUnorm2 } from "../../engine/utils";
import { Meshes, packMeshStreams, type QuantStreams } from "./mesh";
import {
    decodeColor,
    decodeTangent,
    decodeUv1,
    encodeTangent,
    MeshQuant,
    packTangent,
} from "./vertex";

// Oct x has error <= 0.5/32767; sign-biased/remapped y <= 2/32767.
// Unfolding is sqrt(2)-Lipschitz in L1, and normalization costs at most
// 2*sqrt(3), since an oct vector's length is >= 1/sqrt(3).
// 2*sqrt(6)*2.5/32767 plus f32 rounding is less than 0.0004.
const TANGENT_ERROR = 0.0004;
const COLOR_ERROR = 0.5 / 65535 + 2 ** -25;
const normals = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [0.3, -0.4, Math.sqrt(0.75)],
];
test("the tangent CPU encoder mirrors TGSL, including the sign-preserving octahedral seam", () => {
    for (let i = 0; i < 300; i++) {
        const x = Math.sin(i * 2.3),
            y = Math.cos(i * 3.7),
            z = Math.sin(i * 5.1);
        const length = Math.hypot(x, y, z);
        normals.push([x / length, y / length, z / length]);
    }
    for (const [x, y, z] of normals)
        for (const sign of [-1, 1]) {
            const word = packTangent(x, y, z, sign);
            expect(word).toBe(encodeTangent(d.vec4f(x, y, z, sign)));
            const decoded = decodeTangent(word);
            expect(decoded.w).toBe(sign);
            expect(Math.hypot(decoded.x - x, decoded.y - y, decoded.z - z)).toBeLessThan(
                TANGENT_ERROR,
            );
        }
});

test("color words mirror TGSL unorm16 and UV_1 preserves the f32 bits", () => {
    for (let i = 0; i <= 255; i++) {
        const source = new Float32Array([
            i / 255,
            (255 - i) / 255,
            ((i + 73) % 256) / 255,
            ((i + 131) % 256) / 255,
        ]);
        const words = d.vec2u(packUnorm2(source[0], source[1]), packUnorm2(source[2], source[3]));
        expect(words.x).toBe(pack2x16unorm(d.vec2f(source[0], source[1])));
        expect(words.y).toBe(pack2x16unorm(d.vec2f(source[2], source[3])));
        const decoded = decodeColor(words);
        expect([decoded.x, decoded.y, decoded.z, decoded.w]).toEqual([...source]);
        const floats = new Float32Array([i / 256, 0.123456789, 0.987654321, (256 - i) / 256]);
        const quantized = decodeColor(
            d.vec2u(packUnorm2(floats[0], floats[1]), packUnorm2(floats[2], floats[3])),
        );
        expect(
            Math.max(
                ...[quantized.x, quantized.y, quantized.z, quantized.w].map((v, c) =>
                    Math.abs(v - floats[c]),
                ),
            ),
        ).toBeLessThanOrEqual(COLOR_ERROR);
    }
    const floats = new Float32Array([123456.789, -0.000012345]);
    const words = new Uint32Array(floats.buffer);
    const decoded = decodeUv1(d.vec2u(words[0], words[1]));
    expect([decoded.x, decoded.y]).toEqual([...floats]);
});

test("stream families append interleaved absolute-index words and pad only the family end", () => {
    expect(d.sizeOf(MeshQuant)).toBe(48);
    for (let mask = 0; mask < 8; mask++) {
        const staged = [1, 2].map((count) => ({
            vertices: new Float32Array(count * 8),
            ...(mask & 1
                ? { uv1: new Float32Array(Array.from({ length: count * 2 }, (_, i) => 10 + i)) }
                : {}),
            ...(mask & 2
                ? {
                      tangent: new Float32Array(
                          Array.from({ length: count }, () => [1, 0, 0, -1]).flat(),
                      ),
                  }
                : {}),
            ...(mask & 4
                ? {
                      color: new Float32Array(
                          Array.from({ length: count }, () => [0.2, 0.4, 0.6, 0.8]).flat(),
                      ),
                  }
                : {}),
        }));
        const q: QuantStreams = {
            main: new Uint32Array(12).fill(42),
            position: new Uint32Array(6),
            quant: new Float32Array(24),
        };
        const original = q.main;
        const streams = packMeshStreams(staged, q);
        const stride = Number(!!(mask & 1)) * 2 + Number(!!(mask & 2)) + Number(!!(mask & 4)) * 2;
        expect(q.main.length).toBe(12 + Math.ceil((3 * stride) / 4) * 4);
        expect([...q.main.subarray(0, 12)]).toEqual([...original]);
        expect(new Uint32Array(q.quant.buffer)[10]).toBe(mask ? 3 : 0);
        expect(new Uint32Array(q.quant.buffer)[22]).toBe(mask ? 3 : 0);
        if (!mask) {
            expect(streams).toBeUndefined();
            expect(q.main).toBe(original);
            continue;
        }
        let at = 12;
        for (const mesh of staged)
            for (let v = 0; v < mesh.vertices.length / 8; v++) {
                if (mesh.uv1) {
                    const decoded = decodeUv1(d.vec2u(q.main[at++], q.main[at++]));
                    expect([decoded.x, decoded.y]).toEqual([mesh.uv1[v * 2], mesh.uv1[v * 2 + 1]]);
                }
                if (mesh.tangent) {
                    const decoded = decodeTangent(q.main[at++]);
                    expect(Math.hypot(decoded.x - 1, decoded.y, decoded.z)).toBeLessThan(
                        TANGENT_ERROR,
                    );
                    expect(decoded.w).toBe(-1);
                }
                if (mesh.color) {
                    expect(q.main[at++]).toBe(0x66663333);
                    expect(q.main[at++]).toBe(0xcccc9999);
                }
            }
        expect([...q.main.subarray(at)]).toEqual(Array(q.main.length - at).fill(0));
    }
});

for (const stream of ["uv1", "tangent", "color"] as const)
    test(`wrong ${stream} lengths refuse naming mesh and stream`, () => {
        const world = new World();
        try {
            expect(() =>
                world.resource(Meshes).add({
                    name: "wrong-stream",
                    vertices: new Float32Array(24),
                    indices: new Uint32Array([0, 1, 2]),
                    [stream]: new Float32Array(1),
                }),
            ).toThrow(new RegExp(`wrong-stream.*${stream}.*expected`));
        } finally {
            world.dispose();
        }
    });
