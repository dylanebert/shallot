import { expect, test } from "bun:test";
import gold from "../../crates/physics/tests/joint_motion.gold.json";
import { visitJointMotion } from "../../src/standard/physics/solver/joint-motion.fixture";
import { nativeBinary, run } from "./native";
test("joint translation, speed and length equal native Box3D with static, kinematic and sleeping bodies",()=>{
    const native=JSON.parse(run([nativeBinary("joint-motion.c")])) as typeof gold;
    expect(native).toEqual(gold);
    visitJointMotion(native,(actual,expected,label)=>expect(actual,label).toBe(expected));
    console.log(JSON.stringify({cases:native.length,readers:4}));
});
