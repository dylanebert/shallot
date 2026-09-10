import { expect, test } from "bun:test";
import { type TgpuBuffer, writeToArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import {
    type Draw,
    DrawIndexedIndirect,
    type DrawIndirectBuffer,
    validateDrawIndirect,
} from "./registry";

test("Draw carries the producer's indirect-buffer schema", () => {
    const indirect = {
        resourceType: "buffer",
        dataType: DrawIndexedIndirect,
        usableAsIndirect: true,
    } as TgpuBuffer<typeof DrawIndexedIndirect> & { usableAsIndirect: true };
    const draw: Draw = {
        name: "typed-draw",
        surface: "typed-surface",
        mesh: "typed-mesh",
        args: { indirect },
    };

    expect(draw.args.indirect.dataType).toBe(indirect.dataType);
});

test("the canonical indirect record preserves signed baseVertex", () => {
    const bytes = new ArrayBuffer(d.sizeOf(DrawIndexedIndirect));
    writeToArrayBuffer(bytes, DrawIndexedIndirect, {
        indexCount: 3,
        instanceCount: 2,
        firstIndex: 1,
        baseVertex: -17,
        firstInstance: 4,
    });

    expect(new Int32Array(bytes)[3]).toBe(-17);
});

const unrelated = {
    resourceType: "buffer",
    dataType: d.struct({ unrelated: d.arrayOf(d.u32, 5) }),
    usableAsIndirect: true,
} as TgpuBuffer<ReturnType<typeof d.struct<{ unrelated: d.WgslArray<d.U32> }>>> & {
    usableAsIndirect: true;
};

// @ts-expect-error a byte-compatible but unrelated schema is not an indirect draw record
const rejected: DrawIndirectBuffer = unrelated;
void rejected;

test("an unrelated byte-compatible indirect schema is rejected at runtime", () => {
    expect(() => validateDrawIndirect(unrelated as never)).toThrow(/DrawIndexedIndirect schema/);
});
