import { expect, test } from "bun:test";
import { resolve } from "./kernel";

for (const [cores, want] of [
    [0, 1],
    [1, 1],
    [2, 1],
    [10, 5],
    [16, 8],
    [32, 8],
]) {
    test(`default threading uses ${want} threads for ${cores} logical cores`, () => {
        expect(resolve(undefined, { browser: false, shared: true, cores })).toEqual({
            want,
            warn: false,
        });
    });
}

test("explicit thread counts ignore the host's core count", () => {
    expect(resolve(3, { browser: false, shared: true, cores: 32 })).toEqual({
        want: 3,
        warn: false,
    });
    expect(resolve(0, { browser: true, shared: false, cores: 32 })).toEqual({
        want: 0,
        warn: false,
    });
});
