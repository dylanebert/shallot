import { expect, test } from "bun:test";
import { removeTestDirectory } from "./test-directory";

const failure = (code: string) => Object.assign(new Error(code), { code });

test("a lock held past a child's exit is retried after a backoff until removal succeeds", () => {
    let calls = 0;
    const waits: number[] = [];
    removeTestDirectory(
        "held",
        () => {
            if (++calls < 3) throw failure("EBUSY");
        },
        (ms) => waits.push(ms),
    );
    expect(calls).toBe(3);
    expect(waits).toEqual([100, 200]);
});

test("a failure no lock explains is thrown without retrying", () => {
    let calls = 0;
    expect(() =>
        removeTestDirectory("denied", () => {
            calls++;
            throw failure("EACCES");
        }),
    ).toThrow("EACCES");
    expect(calls).toBe(1);
});
