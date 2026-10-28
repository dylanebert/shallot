import { expect, test } from "bun:test";
import { readdirSync, statSync } from "node:fs";
import { digestMismatch, drawingDigest, eventDigests, querySteps, recordDigest } from "./digest";

test("record digests use FNV-1a u64 UTF-8 bytes", () => {
    expect(recordDigest("")).toBe("cbf29ce484222325");
    expect(recordDigest("hello")).toBe("a430d84680aabd0b");
});

test("drawing digests hold every callback field, row boundary and signed zero", () => {
    const records = [
        [3, 0, 1, 2, 0, 0, 0, 1, 0, -0.5, 0, 0, 0.5, 0, 0.35],
        [0, 0, 1, 2, 6],
    ];
    const original = drawingDigest(records);
    for (let row = 0; row < records.length; row++)
        for (let field = 0; field < records[row]!.length; field++) {
            const changed = structuredClone(records);
            changed[row]![field]! += 0.125;
            expect(drawingDigest(changed)).not.toBe(original);
        }
    expect(drawingDigest([[0]])).not.toBe(drawingDigest([[-0]]));
    expect(drawingDigest([[1, 2], [3]])).not.toBe(drawingDigest([[1], [2, 3]]));
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

test("Cast digests hold each invocation including its mover planes", () => {
    const cast = "cast 0 00000000\ncast 0 00000000\noverlap 1\nmover 0\n";
    const planes = cast.replace("mover 0\n", "mover 2\nplane 00000000\nplane 00000001\n");
    expect(querySteps("cast", cast + planes)).toEqual([cast, planes]);
    const changed = planes.replace("plane 00000001", "plane 00000002");
    expect(querySteps("cast", cast + changed).map(recordDigest)).toEqual([
        recordDigest(cast),
        recordDigest(changed),
    ]);
    expect(recordDigest(changed)).not.toBe(recordDigest(planes));
});

test("every physics gold is at most 100 KB", () => {
    for (const file of readdirSync(import.meta.dir).filter((file) => file.endsWith(".gold.txt"))) {
        expect(statSync(`${import.meta.dir}/${file}`).size, file).toBeLessThanOrEqual(100000);
    }
});
