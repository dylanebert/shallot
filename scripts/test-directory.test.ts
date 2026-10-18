import { expect, test } from "bun:test";
import { removeTestDirectory } from "./test-directory";

const failure = (code: string) => Object.assign(new Error(code), { code });

test("a lock held past a child's exit is retried until removal succeeds", () => {
    let calls = 0;
    removeTestDirectory("held", () => {
        if (++calls === 1) throw failure("EBUSY");
    });
    expect(calls).toBe(2);
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
