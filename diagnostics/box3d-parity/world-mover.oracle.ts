import { expect, setDefaultTimeout, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";

setDefaultTimeout(0);
test("world mover gold is native traversal and capacity", async () => {
    expect(nativeSseOutput("world-mover.c", "")).toBe(await Bun.file(new URL("../../src/standard/physics/world-mover.gold.txt", import.meta.url)).text());
});
