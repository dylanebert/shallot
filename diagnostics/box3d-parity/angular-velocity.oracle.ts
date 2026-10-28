import { expect, test } from "bun:test";
import gold from "../../src/standard/physics/angular-velocity.gold.json";
import { nativeBinary, run } from "./native";

test("angular velocity gold is native Box3D's current value", () => {
    expect(JSON.parse(run([nativeBinary("angular-velocity.c")]))).toEqual(gold);
});
