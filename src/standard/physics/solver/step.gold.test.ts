// Active scene parity runs the immutable official command/bundle corpus. The historical 53 scene
// fixtures remain under fixtures/ as migration evidence, but their predecessor hashes are not the
// current authority and are intentionally not asserted here.

import { expect } from "bun:test";
import { check } from "../../../harness/check";
import { World } from "../api/world";
import { BodyType } from "../common/types";
import { loadConsumerCorpus, runCommonInput } from "../oracle/consumer";
import { loadScenarioCorpus, runScenario } from "../oracle/scenario";
import { compareCase } from "../oracle/strict";

check(
    "CCD and sensor manifold intermediates match the active C route",
    {
        claim: "the active collision route changes the symmetric face-B feature order, the pinned CCD and sensor intermediate bits, or the public body move record identity",
        size: "integration",
        budget: 20_000,
        subject: ["crates/physics", "src/standard/physics/collision"],
    },
    () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 1, z: 0 } });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        body.applyMassFromShapes();
        world.step(1 / 60, 1);
        const moves = world.getBodyEvents();
        expect(moves.count).toBeGreaterThan(0);
        const event = moves.moveEvents[0];
        expect(event.body.id.index1).toBe(body.id.index1);
        expect(event.body.id.generation).toBe(body.id.generation);
        expect(event.fellAsleep).toBe(false);

        const { corpus, digest } = loadScenarioCorpus();
        const ccd = corpus.scenarios.find((scenario) => scenario.name === "ccd-bullet");
        const sensor = corpus.scenarios.find((scenario) => scenario.name === "sensor");
        if (!ccd || !sensor) throw new Error("required CCD/sensor scenarios are missing");
        const bits = (value: number): string => {
            const view = new DataView(new ArrayBuffer(4));
            view.setFloat32(0, value);
            return view.getUint32(0).toString(16).padStart(8, "0");
        };
        const features = [0x0112010b, 0x01070112, 0x01100107, 0x010b0110];
        let ccdChecked = false;
        const ccdResult = runScenario(ccd, digest, false, (world, step) => {
            if (step !== 0) return;
            const manifold = world.state.contacts[0]?.manifolds[0];
            if (!manifold) throw new Error("CCD step 0 has no contact manifold");
            expect(manifold.points.slice(0, 4).map((point) => point.featureId)).toEqual(features);
            expect(bits(manifold.points[0].anchorA.x)).toBe("3d4ccc00");
            ccdChecked = true;
        });
        expect(ccdChecked).toBe(true);
        expect(ccdResult.hashes.find((hash) => hash.step === 0)?.value).toBe("0xd4475ed2139c0e45");

        let sensorChecked = false;
        const sensorResult = runScenario(sensor, digest, false, (world, step) => {
            if (step !== 14) return;
            const manifold = world.state.contacts[0]?.manifolds[0];
            if (!manifold) throw new Error("sensor step 14 has no face-B contact manifold");
            expect(bits(manifold.normal.x)).toBe("00000000");
            expect(bits(manifold.normal.y)).toBe("3f800000");
            expect(bits(manifold.normal.z)).toBe("00000000");
            expect(manifold.points.slice(0, 4).map((point) => point.featureId)).toEqual(features);
            expect(bits(manifold.points[0].separation)).toBe("3ba3d800");
            sensorChecked = true;
        });
        expect(sensorChecked).toBe(true);
        expect(sensorResult.hashes.find((hash) => hash.step === 14)?.value).toBe(
            "0x31240e262d32461b",
        );
    },
);

check(
    "official Box3D scene command/bundle parity",
    {
        claim: "all 53 official Box3D scenes execute through the immutable command corpus and match its outputs",
        size: "integration",
        budget: 20_000,
    },
    () => {
        const corpus = loadConsumerCorpus();
        const scenes = corpus.cases.filter((item) => item.family === "scenario");
        if (scenes.length !== 53)
            throw new Error(`expected 53 official scenes, got ${scenes.length}`);
        for (const scene of scenes) {
            const result = compareCase(scene, runCommonInput(scene));
            if (result.status !== "pass")
                throw new Error(`${scene.id}: ${result.firstDifference?.path ?? "scene mismatch"}`);
        }
        console.log(JSON.stringify({ corpus: "immutable box3d v6", scenes: scenes.length }));
    },
);
