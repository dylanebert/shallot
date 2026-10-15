import { expect, test } from "bun:test";
import gold from "../../crates/physics/tests/reader_regressions.gold.json";
import { nativeBinary, run } from "./native";

test("body readers and default material mixing gold regenerate from native Box3D", () => {
    expect(JSON.parse(run([nativeBinary("reader-regressions.c")]))).toEqual(gold);
});
