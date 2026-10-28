import { expect, test } from "bun:test";
import gold from "../../src/standard/physics/mass-data.gold.json";
import { nativeBinary, run } from "./native";

test("mass override snapshots retain native Box3D hashes, mass and velocity", () => {
    expect(JSON.parse(run([nativeBinary("mass-data.c")]))).toEqual(gold);
});
