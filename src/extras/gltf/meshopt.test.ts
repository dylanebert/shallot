import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import {
    assertDecoded,
    filters,
    loadRecords,
    modes,
} from "../../../diagnostics/meshopt-records/records";
import { decodeMeshoptView } from "./kernel";

const recordedViews = await loadRecords();
test("unwritten FIFOs preserve the pinned decoder's sentinel bits", () => {
    const records = readFileSync(
        new URL("../../../crates/gltf/tests/meshopt/fifo.bin", import.meta.url),
    );
    let offset = 0;
    for (const byteStride of [2, 4]) {
        for (let record = 0; record < 3; record++) {
            const source = records.subarray(offset, offset + 18);
            offset += 18;
            const expected = records.subarray(offset, offset + 3 * byteStride);
            offset += expected.length;
            const output = new Uint8Array(expected.length);
            decodeMeshoptView(
                42,
                { count: 3, byteStride, mode: "TRIANGLES" },
                source,
                output,
                false,
            );
            expect(Buffer.from(output).equals(expected)).toBe(true);
        }
    }
    expect(offset).toBe(records.length);
});

for (const record of recordedViews) {
    const {
        asset,
        index,
        count,
        stride: byteStride,
        mode,
        filter,
        khr,
        source,
        length: decoded,
    } = record;
    const view = { count, byteStride, mode: modes[mode], filter: filters[filter] };
    test(`${asset} bufferView ${index} is bit-equal through the SIMD JavaScript entry`, async () => {
        const output = new Uint8Array(decoded);
        decodeMeshoptView(index, view, source, output, khr !== 0);
        await assertDecoded(record, output);
        if (mode === 0 && (source[0] === 0xa1 || filter === 4)) {
            expect(() => decodeMeshoptView(index, view, source, output, false)).toThrow(
                `bufferView ${index}:`,
            );
        }
        if (mode !== 0) {
            for (const invalid of ["OCTAHEDRAL", "QUATERNION", "EXPONENTIAL", "COLOR", "UNKNOWN"]) {
                expect(() =>
                    decodeMeshoptView(
                        index,
                        { ...view, filter: invalid },
                        source,
                        output,
                        khr !== 0,
                    ),
                ).toThrow(`bufferView ${index}:`);
            }
        }
        expect(() =>
            decodeMeshoptView(index, { ...view, mode: "UNKNOWN" }, source, output, khr !== 0),
        ).toThrow(`bufferView ${index}:`);
    });
    test(`${asset} bufferView ${index} rejects malformed streams without a WASM trap`, () => {
        const output = new Uint8Array(decoded);
        for (const length of [0, 1, Math.floor(source.length / 2), source.length - 1]) {
            expect(() =>
                decodeMeshoptView(index, view, source.subarray(0, length), output, khr !== 0),
            ).toThrow(`bufferView ${index}:`);
        }
        const corrupt = Uint8Array.from(source);
        corrupt[0] = 0;
        expect(() => decodeMeshoptView(index, view, corrupt, output, khr !== 0)).toThrow(
            `bufferView ${index}:`,
        );
    });
}

for (let byteStride = 4; byteStride <= 256; byteStride += 4) {
    test(`SIMD attribute blocks and tails at stride ${byteStride}`, () => {
        for (const count of [1, 16, 17, 257, 513]) {
            for (const version of [0, 1]) {
                const blockSize = Math.min(Math.floor(8192 / byteStride) & ~15, 256);
                const source = [0xa0 + version];
                for (let offset = 0; offset < count; offset += blockSize) {
                    const n = Math.min(count - offset, blockSize);
                    source.push(
                        ...Array(
                            version
                                ? byteStride / 4
                                : Math.ceil(Math.ceil(n / 16) / 4) * byteStride,
                        ).fill(version ? 0xaa : 0),
                    );
                }
                const tail = byteStride + (version ? byteStride / 4 : 0);
                source.push(...Array(Math.max(tail, version ? 24 : 32) - tail).fill(0));
                source.push(...Array.from({ length: byteStride }, (_, i) => i));
                if (version) source.push(...Array(byteStride / 4).fill(0));
                const output = new Uint8Array(count * byteStride);
                const expected = Uint8Array.from(
                    { length: output.length },
                    (_, i) => i % byteStride,
                );
                const view = { count, byteStride, mode: "ATTRIBUTES" };
                decodeMeshoptView(78, view, Uint8Array.from(source), output, true);
                expect(Buffer.from(output).equals(Buffer.from(expected))).toBe(true);
                source.push(0);
                expect(() =>
                    decodeMeshoptView(78, view, Uint8Array.from(source), output, true),
                ).toThrow("bufferView 78:");
            }
        }
    });
}

test("SIMD prefix scans wrap byte/halfword sums and carry rotated XOR across blocks", () => {
    const count = 513,
        byteStride = 12;
    const source = [0xa1];
    for (let offset = 0; offset < count; offset += 256) {
        const n = Math.min(count - offset, 256);
        source.push(0xff, 0xff, 0xff);
        for (let channel = 0; channel < 3; channel++) {
            for (const encoded of [1, 3, 5, 7]) source.push(...Array(n).fill(encoded));
        }
    }
    // 12 baseline bytes + three channel bytes, preceded by padding to the 24-byte tail.
    source.push(...Array(9).fill(0), ...Array(12).fill(0), 0, 1, 0x52);
    const output = new Uint8Array(count * byteStride);
    decodeMeshoptView(
        79,
        { count, byteStride, mode: "ATTRIBUTES" },
        Uint8Array.from(source),
        output,
        true,
    );
    const expected = new Uint8Array(output.length);
    const values = new DataView(expected.buffer);
    const rotated = (0x07050301 >>> 5) | (0x07050301 << 27);
    for (let i = 0; i < count; i++) {
        for (let j = 0; j < 4; j++) expected[i * 12 + j] = -(j + 1) * (i + 1);
        values.setUint16(i * 12 + 4, -385 * (i + 1), true);
        values.setUint16(i * 12 + 6, -899 * (i + 1), true);
        values.setUint32(i * 12 + 8, (i + 1) % 2 ? rotated : 0, true);
    }
    expect(Buffer.from(output).equals(Buffer.from(expected))).toBe(true);
});

test("the JavaScript entry refuses counts that would truncate at the WASM boundary", () => {
    for (const count of [-1, 0, 0x100000001, Number.NaN, 1.5]) {
        expect(() =>
            decodeMeshoptView(
                77,
                { count, byteStride: 4, mode: "ATTRIBUTES" },
                new Uint8Array(),
                new Uint8Array(4),
                true,
            ),
        ).toThrow("bufferView 77:");
    }
});
