import { expect, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";

test("mover public-query gold is native inverse body data and capsule casts", async () => {
    expect(nativeSseOutput("mover-public.c", "")).toBe(await Bun.file(new URL("../../src/standard/physics/mover-public.gold.txt", import.meta.url)).text());
});
