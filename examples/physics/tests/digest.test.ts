import { expect, test } from "bun:test";
import { readdirSync, statSync } from "node:fs";
import { digestMismatch, eventDigests, querySteps, recordDigest } from "./digest";

test("record digests use FNV-1a u64 UTF-8 bytes", () => {
    expect(recordDigest("")).toBe("cbf29ce484222325");
    expect(recordDigest("hello")).toBe("a430d84680aabd0b");
});

test("one event field changes only its step digest", () => {
    const original = "0000000000000000\n0000000000000001 [[[1,2,3,4]],[],[],[],[],[]]\n";
    const a = eventDigests(original);
    const b = eventDigests(original.replace("1,2,3,4", "1,2,3,5"));
    expect(a[0]).toBe(b[0]);
    expect(a[1]).not.toBe(b[1]);
    expect(eventDigests(original.replace("0000000000000001", "0000000000000002"))[1]).not.toBe(
        a[1],
    );
    expect(eventDigests("0000000000000000\n")).toEqual(
        eventDigests("0000000000000000 [[],[],[],[],[],[]]\n"),
    );
});

test("each cast field and hull vertex is visible in its step digest", () => {
    const cast = "cast 1 00000000 00000000 00000000 00000000 00000000 00000000 00000000\n";
    const steps = querySteps("heightField", cast.repeat(28556));
    expect(steps).toEqual([cast.repeat(14278), cast.repeat(14278)]);
    const expected = [cast, cast].map(recordDigest);
    for (let field = 1; field <= 8; field++) {
        const fields = cast.trimEnd().split(" ");
        fields[field] = field === 1 ? "0" : "00000001";
        const changed = [cast, `${fields.join(" ")}\n`].map(recordDigest);
        expect(changed[0]).toBe(expected[0]);
        expect(changed[1]).not.toBe(expected[1]);
    }
    const hull = "hull 3f800000 00000000 00000000 00000000\n";
    expect(recordDigest(querySteps("hull", hull)[0]!)).not.toBe(
        recordDigest(querySteps("hull", hull.replace("00000000", "00000001"))[0]!),
    );
    expect(digestMismatch("heightField", 1)).toContain(
        "bun diagnostics/box3d-parity/records.ts heightField 1",
    );
});

test("every physics gold is at most 100 KB", () => {
    for (const file of readdirSync(import.meta.dir).filter((file) => file.endsWith(".gold.txt"))) {
        expect(statSync(`${import.meta.dir}/${file}`).size, file).toBeLessThanOrEqual(100000);
    }
});
