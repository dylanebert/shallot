import { afterEach, expect, test } from "bun:test";
import type { TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Compute } from "../../engine";
import { mirror } from ".";

afterEach(() => {
    Object.assign(Compute, { root: undefined });
});

test("Mirror retains the schema-carrying source while copying from its raw twin", () => {
    const raw = { size: 16 } as GPUBuffer;
    const source = {
        resourceType: "buffer",
        dataType: d.arrayOf(d.u32, 4),
    } as TgpuBuffer<d.WgslArray<d.U32>>;
    Object.assign(Compute, {
        root: { unwrap: (value: unknown) => (value === source ? raw : value) },
    });

    const m = mirror(source);
    expect(m.source).toBe(source);
    expect(m.size).toBe(16);
    m.dispose();
});
