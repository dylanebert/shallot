import { describe, expect, test } from "bun:test";
import { clz32, lowerPowerOf2Exponent } from "./bits";
import {
    countSetBits,
    createBitSet,
    getBit,
    inPlaceUnion,
    setBit,
    setBitCountAndClear,
    setBitGrow,
} from "./bitset";

// Ports test_bitset.c's TestBitMath.
describe("bit math", () => {
    test("clz32", () => {
        expect(clz32(9)).toBe(31 - 3);
    });

    test("lowerPowerOf2Exponent equals floor(log2)", () => {
        for (let i = 1; i < 1000; ++i) {
            // Independent reference: bit-length - 1 is the exact floor(log2(i)).
            const expected = i.toString(2).length - 1;
            expect(lowerPowerOf2Exponent(i)).toBe(expected);
        }
    });
});

// Ports test_bitset.c's TestBitSet: a Fibonacci-indexed bit pattern.
describe("bit set", () => {
    const Count = 169;

    test("set/get over a Fibonacci pattern", () => {
        const bitSet = createBitSet(Count);
        setBitCountAndClear(bitSet, Count);
        const values = new Array<boolean>(Count).fill(false);

        let i1 = 0;
        let i2 = 1;
        setBit(bitSet, i1);
        values[i1] = true;
        while (i2 < Count) {
            setBit(bitSet, i2);
            values[i2] = true;
            const next = i1 + i2;
            i1 = i2;
            i2 = next;
        }

        for (let i = 0; i < Count; ++i) {
            expect(getBit(bitSet, i)).toBe(values[i]);
        }
    });

    test("setBitGrow extends past the initial block count", () => {
        const bitSet = createBitSet(8);
        setBitCountAndClear(bitSet, 8);
        expect(getBit(bitSet, 500)).toBe(false);
        setBitGrow(bitSet, 500);
        expect(getBit(bitSet, 500)).toBe(true);
        expect(getBit(bitSet, 499)).toBe(false);
    });

    test("countSetBits and inPlaceUnion", () => {
        const a = createBitSet(128);
        const b = createBitSet(128);
        setBitCountAndClear(a, 128);
        setBitCountAndClear(b, 128);

        for (const i of [1, 40, 63, 64, 127]) {
            setBit(a, i);
        }
        for (const i of [40, 64, 100]) {
            setBit(b, i);
        }
        expect(countSetBits(a)).toBe(5);
        expect(countSetBits(b)).toBe(3);

        inPlaceUnion(a, b);
        // Union of {1,40,63,64,127} and {40,64,100} = {1,40,63,64,100,127} = 6 distinct.
        expect(countSetBits(a)).toBe(6);
        expect(getBit(a, 100)).toBe(true);
    });
});
