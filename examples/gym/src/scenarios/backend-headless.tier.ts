import { describe, expect, test } from "bun:test";
import { Body, build, Compute, MirrorPlugin, Physics, SlabPlugin } from "@dylanebert/shallot";
import { AvbdPlugin } from "@dylanebert/shallot/avbd";
import {
    BACKEND_BOX_HALF,
    BACKEND_DROP_Y,
    BACKEND_FALL_THROUGH_BAND,
    BACKEND_FLOOR_HALF_Y,
    BACKEND_REST_BAND,
    BACKEND_REST_Y,
} from "./backend-geometry";

// Trigger cone: `examples/gym/src/scenarios/backend-headless.tier.ts`, `backend-geometry.ts`, and the
// published physics substrate/backend modules it imports. Run from the shallot root with
// `bun test ./examples/gym/src`. The verdict derives its settle and no-fall-through bands from the
// shared authored geometry constants imported below; the browser twin remains the reference.

const CAPACITY = 8192;
const TICKS = 240;

describe("headless backend settle verdict", () => {
    test("settles without falling through in the browser twin's authored band", async () => {
        const app = await build({
            plugins: [SlabPlugin, MirrorPlugin, AvbdPlugin],
            defaults: false,
            capacity: CAPACITY,
            scene: `<scene>
                <a body="pos: 0 0 0; half-extents: 10 ${BACKEND_FLOOR_HALF_Y} 10; mass: 0" />
                <a body="pos: 0 ${BACKEND_DROP_Y} 0; half-extents: ${BACKEND_BOX_HALF} ${BACKEND_BOX_HALF} ${BACKEND_BOX_HALF}; mass: 1" />
            </scene>`,
        });
        try {
            const box = [...app.state.query([Body])].find((eid) => Body.mass.get(eid) > 0);
            expect(box).toBeDefined();
            if (box !== undefined) Physics.backend?.readBody(box);
            for (let tick = 0; tick < TICKS; tick++) {
                app.state.step();
                await Compute.device.queue.onSubmittedWorkDone();
                await Bun.sleep(0);
            }

            const pose = box === undefined ? null : Physics.backend?.readBody(box);
            expect(pose).not.toBeNull();
            const y = pose?.pos[1] ?? Number.NaN;
            expect(Number.isFinite(y), "settled body y is finite").toBe(true);
            expect(y, "body did not fall through the floor").toBeGreaterThan(
                BACKEND_REST_Y - BACKEND_FALL_THROUGH_BAND,
            );
            expect(
                y,
                `body rests at ${BACKEND_REST_Y} ± ${BACKEND_REST_BAND} from shared authored geometry`,
            ).toBeWithin(BACKEND_REST_Y - BACKEND_REST_BAND, BACKEND_REST_Y + BACKEND_REST_BAND);
        } finally {
            app.dispose();
        }
    });
});
