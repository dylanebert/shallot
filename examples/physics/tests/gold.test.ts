import { expect, test } from "bun:test";
import { sampleGold } from "./gold";

test("sample gold retains all 64 hash bits and reconstructs omitted empty events independently", () => {
    const records = sampleGold("ffffffffffffffff\n0000000000000001\n");
    expect(records.map((record) => record.hash)).toEqual([0xffffffffffffffffn, 1n]);
    expect(records.map((record) => record.events)).toEqual([
        [[], [], [], [], [], []],
        [[], [], [], [], [], []],
    ]);
    records[0]!.events[0]!.push([1]);
    expect(records[1]!.events[0]).toEqual([]);
});

test("sample gold reconstructs event records without losing float or material-id data", () => {
    const events = [
        [],
        [],
        [],
        [],
        [[1, 2, 3, 4, 0.125, -0.25, 3, 0, 1, 0, 4.5, "18446744073709551615", "42"]],
        [],
    ];
    expect(sampleGold(`0123456789abcdef ${JSON.stringify(events)}\n`)[0]).toEqual({
        hash: 0x0123456789abcdefn,
        events,
    });
});

test("bad sample gold names its step and whether its line or event records are invalid", () => {
    expect(() => sampleGold("0000000000000000\nbad\n")).toThrow("step 1: invalid sample gold line");
    for (const events of ["[", "[]", "[[],[],[],[],[],null]"])
        expect(() => sampleGold(`0000000000000000 ${events}\n`)).toThrow(
            "step 0: invalid sample event records",
        );
});
