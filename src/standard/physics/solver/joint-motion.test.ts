import { expect, test } from "bun:test";
import gold from "../../../../crates/physics/tests/joint_motion.gold.json";
import { visitJointMotion } from "./joint-motion.fixture";

test("joint translation, speed and length readers equal native Box3D bits", () => {
    visitJointMotion(gold, (actual, expected, label) => expect(actual, label).toBe(expected));
});
