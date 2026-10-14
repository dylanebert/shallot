import { expect, test } from "bun:test";
import gold from "../../crates/physics/tests/joint_read.gold.json";
import { visitJointReaders } from "../../src/standard/physics/solver/joint-read.fixture";
import { nativeBinary, run } from "./native";

test("joint rotations and separation errors equal native Box3D", () => {
    const native=JSON.parse(run([nativeBinary("joint-read.c")])) as typeof gold;
    expect(native).toEqual(gold);
    visitJointReaders(native,(actual,expected,label)=>expect(actual,label).toBe(expected));
    console.log(JSON.stringify({cases:native.length,readers:6}));
});
