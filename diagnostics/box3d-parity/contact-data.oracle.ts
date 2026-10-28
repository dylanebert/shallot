import { expect, test } from "bun:test";
import gold from "../../src/standard/physics/contact-data.gold.json";
import { nativeBinary, run } from "./native";

test("contact data gold equals every field of native Box3D manifolds", () => {
    expect(JSON.parse(run([nativeBinary("contact-data.c")]))).toEqual(gold);
});
