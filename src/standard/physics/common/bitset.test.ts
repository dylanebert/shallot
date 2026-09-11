import { expect } from "bun:test";
import { check } from "../../../harness/check";
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

check(
    "clz32 counts leading zeros the way the C intrinsic does",
    {
        claim: "the physics clz32 binding is off by one against the bit position it reports for a small operand",
        tier: "step",
    },
    () => {
        expect(clz32(9)).toBe(31 - 3);
    },
);

check(
    "lowerPowerOf2Exponent equals floor of log2",
    {
        claim: "lowerPowerOf2Exponent is off by one at or near a power of two, so a broadphase bucket sizes to the wrong exponent",
        tier: "step",
    },
    () => {
        for (let i = 1; i < 1000; ++i) {
            // Independent reference: bit-length - 1 is the exact floor(log2(i)).
            const expected = i.toString(2).length - 1;
            expect(lowerPowerOf2Exponent(i)).toBe(expected);
        }
    },
);

// Ports test_bitset.c's TestBitSet: a Fibonacci-indexed bit pattern.

check(
    "setBit and getBit agree over a Fibonacci pattern",
    {
        claim: "the bit set indexes the wrong 32-bit block or shift, so a Fibonacci-indexed pattern reads back with bits in the wrong slots",
        tier: "step",
    },
    () => {
        const Count = 169;
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
    },
);

check(
    "setBitGrow extends past the initial block count",
    {
        claim: "setBitGrow fails to grow the backing blocks for a far index, or clobbers its neighbour while growing",
        tier: "step",
    },
    () => {
        const bitSet = createBitSet(8);
        setBitCountAndClear(bitSet, 8);
        expect(getBit(bitSet, 500)).toBe(false);
        setBitGrow(bitSet, 500);
        expect(getBit(bitSet, 500)).toBe(true);
        expect(getBit(bitSet, 499)).toBe(false);
    },
);

check(
    "countSetBits and inPlaceUnion agree on set algebra",
    {
        claim: "countSetBits miscounts across block boundaries or inPlaceUnion double-counts the overlap of two bit sets",
        tier: "step",
    },
    () => {
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
    },
);
