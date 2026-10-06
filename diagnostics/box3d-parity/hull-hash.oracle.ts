import { expect, test } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHull, makeBoxHull, makeOffsetBoxHull } from "../../src/standard/physics/shapes/hull";
import { hash64NonZero, hullImage } from "../../src/standard/physics/shapes/hullbytes";
import { nativeBinary, run } from "./native";

test("hull byte images and full 64-bit hashes equal direct native box construction", () => {
    nativeBinary();
    const box3d = process.env.BOX3D!;
    const output = join(tmpdir(), "shallot-hull-hash-native");
    run(["cc", "-O2", "-std=c17", "-ffp-contract=off", `-I${join(box3d, "include")}`,
        `-I${join(box3d, "src")}`, join(import.meta.dir, "hull-hash.c"),
        join(tmpdir(), "box3d-parity-47d7f7cc/cmake/src/libbox3d.a"), "-o", output]);
    const result = Bun.spawnSync([output]);
    expect(result.exitCode).toBe(0);
    expect(result.stderr.toString()).toBe("miss=1 hit=0 destroy=0\n");
    const bytes = new Uint8Array(result.stdout);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const hulls = [makeBoxHull(1, 1, 1), makeBoxHull(0.5, 1, 2),
        makeOffsetBoxHull(0.75, 1.25, 2.5, { x: 1, y: -2, z: 3 }),
        createHull([{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 },
            { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 }], 4)!];
    let offset = 0;
    for (const hull of hulls) {
        const byteCount = view.getUint32(offset + 140, true);
        const nativeHash = view.getBigUint64(offset + 8, true);
        const native = bytes.slice(offset, offset + byteCount);
        const image = hullImage(hull);
        expect(image).toEqual(native);
        expect(hull.hash).toBe(nativeHash);
        image[image.length - 1] ^= 1;
        expect(() => expect(image).toEqual(native)).toThrow();
        image[image.length - 1] ^= 1;
        expect(() => expect(hull.hash ^ 1n).toBe(nativeHash)).toThrow();
        image.fill(0, 8, 16);
        expect(hash64NonZero(image)).toBe(nativeHash);
        image[image.length - 1] ^= 1;
        expect(hash64NonZero(image)).not.toBe(nativeHash);
        offset += byteCount;
    }
    expect(offset).toBe(bytes.length);
});
