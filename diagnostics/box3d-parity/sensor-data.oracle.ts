import { expect, test } from "bun:test";
import gold from "../../src/standard/physics/sensor-data.gold.json";
import { nativeBinary, run } from "./native";

test("sensor visitor gold equals native Box3D order across begin and end", () => {
    expect(JSON.parse(run([nativeBinary("sensor-data.c")]))).toEqual(gold);
});
