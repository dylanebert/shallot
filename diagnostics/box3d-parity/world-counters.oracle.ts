import { expect, setDefaultTimeout, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";

setDefaultTimeout(0);
test("world counters gold is b3World_GetCounters on ordinary and graph-overflow piles", async () => {
    expect(nativeSseOutput("world-counters.c", "")).toBe(await Bun.file(new URL("../../src/standard/physics/world-counters.gold.txt", import.meta.url)).text());
});
