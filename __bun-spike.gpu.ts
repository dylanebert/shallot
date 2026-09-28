import { expect, test } from "bun:test";

test("Bun runs a non-default test suffix when its path is named", () => {
    expect(2 + 2).toBe(4);
});
