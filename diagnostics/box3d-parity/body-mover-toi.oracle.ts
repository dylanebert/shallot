import { expect, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";

test("body mover TOI gold records native convex hits, misses, initial overlap, filters and both sweeps", async () => {
    expect(nativeSseOutput("body-mover-toi.c", "")).toBe(await Bun.file(new URL("../../src/standard/physics/body-mover-toi.gold.txt", import.meta.url)).text());
});
