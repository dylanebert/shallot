import { expect, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";

test("body mover capacity gold records native counts, attachment order and every plane field", async () => {
    expect(nativeSseOutput("body-query-capacity.c", "")).toBe(await Bun.file(new URL("../../src/standard/physics/body-query-capacity.gold.txt", import.meta.url)).text());
});
