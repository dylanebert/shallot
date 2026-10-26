import { expect, test } from "bun:test";
import { compareRecords } from "./record-diff";

test("one changed event reports its record index and both values", () => {
    const native = "0000000000000001 [[[1,2,3,4],[5,6,7,8]],[],[],[],[],[]]\n";
    const result = compareRecords("benchmarkSensor", 68, native, native.replace("5,6,7,8", "5,6,7,9"));
    expect(result.equal).toBe(false);
    expect(result.message).toBe("benchmarkSensor step 68 record 2 differs\nnative: channel 0 [5,6,7,8]\nShallot: channel 0 [5,6,7,9]");
});

for (const [scene, record] of [
    ["heightField", "cast 1 00000000 00000000 00000000 00000000 00000000 00000000 3f800000"],
    ["hull", "hull 3f800000 00000000 00000000 00000000"],
]) {
    test(`one changed ${scene} field reports its record index and both values`, () => {
        const changed = record!.replace("00000000", "00000001");
        const result = compareRecords(scene!, 0, `${record}\n${record}\n`, `${record}\n${changed}\n`);
        expect(result.equal).toBe(false);
        expect(result.message).toBe(`${scene} step 0 record 1 differs\nnative: ${record}\nShallot: ${changed}`);
        expect(compareRecords(scene!, 0, `${record}\n`, `${record}\n`)).toEqual({ equal: true, message: `${scene} step 0: 1 records equal` });
    });
}

test("a missing record reports both counts and the first missing index", () => {
    expect(compareRecords("hull", 0, "hull a\nhull b\n", "hull a\n")).toEqual({
        equal: false,
        message: "hull step 0 record 1 differs\ncounts: native 2, Shallot 1\nnative: hull b\nShallot: <missing>",
    });
});
