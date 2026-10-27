import { expect, test } from "bun:test";
import publicGold from "../../src/standard/physics/joint-read.gold.json";
import gold from "../../crates/physics/tests/joint_read.gold.json";
import { visitJointReaders } from "../../src/standard/physics/solver/joint-read.fixture";
import { nativeBinary, run } from "./native";

test("joint rotations and separation errors equal native Box3D", () => {
    const native=JSON.parse(run([nativeBinary("joint-read.c")])) as typeof gold;
    expect(native).toEqual(gold);
    expect(JSON.parse(run([nativeBinary("joint-read.c"), "public"]))).toEqual(publicGold);
    visitJointReaders(native,(actual,expected,label)=>expect(actual,label).toBe(expected));
    console.log(JSON.stringify({cases:native.length,readers:6}));
});
