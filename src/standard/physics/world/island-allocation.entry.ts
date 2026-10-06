import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import { ContactField, contactField } from "../collision/contact";
import { SetType } from "../common/constants";
import { BodyType } from "../common/types";
import { addIslandBody, islandArrayCount, removeIslandBody } from "../kernel/islandcolumns";
import { makeBoxHull } from "../shapes/hull";
import { createIsland, linkContact, unlinkContact } from "./island";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const physics = physicsWorld(app.world)!;
    physics.setGravity({ x: 0, y: 0, z: 0 });
    const hull = makeBoxHull(0.5, 0.5, 0.5);
    const bodies = [0, 0.9, 1.8].map((x) => {
        const body = physics.createBody({ type: BodyType.Dynamic, position: { x, y: 0, z: 0 } });
        body.createHull({ density: 1 }, hull);
        return body;
    });
    physics.createDistanceJoint(bodies[0], bodies[1], { length: 0.9, collideConnected: true });
    app.world.step(1 / 60);
    const state = physics.state;
    const ids = bodies.map((body) => body.id.index1 - 1);
    const contacts = state.awakeContacts;
    const find = (a: number, b: number) =>
        contacts.find((id) => {
            const first = contactField(state, id, ContactField.bodyIdA);
            const second = contactField(state, id, ContactField.bodyIdA + 3);
            return (first === a && second === b) || (first === b && second === a);
        });
    const inside = find(ids[0], ids[1]);
    const bridge = find(ids[1], ids[2]);
    if (inside === undefined || bridge === undefined)
        throw new Error("island churn needs both touching contacts");
    const body = state.bodies[ids[2]];
    return {
        // Replay touch-end/begin topology changes without adding collision or solver allocation sites.
        // The permanent joint keeps the first contact inside one island; the bridge merges a singleton.
        step: () => {
            for (let i = 0; i < 32; ++i) {
                unlinkContact(state, inside);
                linkContact(state, inside);
                unlinkContact(state, bridge);
                removeIslandBody(state, body.islandId, body.islandIndex);
                addIslandBody(state, createIsland(state, SetType.Awake), body.id);
                linkContact(state, bridge);
                if (
                    islandArrayCount(state, body.islandId, 0) !== 3 ||
                    contactField(state, inside, ContactField.islandId) !== body.islandId ||
                    contactField(state, bridge, ContactField.islandId) !== body.islandId
                )
                    throw new Error("contact churn lost its merged island");
            }
        },
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
