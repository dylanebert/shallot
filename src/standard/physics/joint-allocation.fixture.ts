import type { PhysicsWorld } from "./api/world";
import { bufferMove } from "./collision/broadphase";
import { contactCount } from "./collision/contact";
import { updateBroadPhasePairs } from "./collision/pairs";
import { BodyType } from "./common/types";
import { DJ_LENGTH, J_EVENT } from "./kernel/columns";
import { EventKind, eventCount } from "./kernel/eventbuffers";
import { collectJointEvents, readJointFloat, writeJointFloat } from "./kernel/jointcolumns";
import { JointField, jointDrawScale, jointField, setJointDrawScale } from "./kernel/jointrecords";
import { kernel } from "./kernel/kernel";
import { ShapeField, shapeField } from "./kernel/shaperecords";
import {
    createJointRecord,
    defaultJointDef,
    destroyJointInternal,
    JointType,
} from "./solver/joint";
import type { WorldState } from "./world/world";

function tune(world: WorldState, id: number, i: number): void {
    const length = i + 0.125;
    const scale = i + 0.5;
    writeJointFloat(world, id, DJ_LENGTH, length);
    setJointDrawScale(world, id, scale);
    if (readJointFloat(world, id, DJ_LENGTH) !== length || jointDrawScale(world, id) !== scale)
        throw new Error("joint allocation subject lost its scalar tuning");
}

export function jointAllocationSubject(physics: PhysicsWorld, control?: () => void): () => void {
    const state = physics.state;
    const defs: ReturnType<typeof defaultJointDef>[] = [];
    const proxies: number[] = [];
    for (let i = 0; i < 32; ++i) {
        const a = physics.createBody({
            type: BodyType.Dynamic,
            position: { x: i * 4, y: 0, z: 0 },
        });
        const b = physics.createBody({
            type: BodyType.Dynamic,
            position: { x: i * 4, y: 0, z: 0 },
        });
        for (const body of [a, b]) {
            const shape = body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
            proxies.push(shapeField(state, shape.id.index1 - 1, ShapeField.proxyKey));
        }
        physics.createFilterJoint(a, b);
        const def = defaultJointDef();
        def.bodyIdA = a.id.index1 - 1;
        def.bodyIdB = b.id.index1 - 1;
        def.userData = i;
        defs.push(def);
    }
    return () => {
        for (let i = 0; i < defs.length; ++i) {
            const id = createJointRecord(state, defs[i], JointType.Distance);
            tune(state, id, i);
            if (jointField(state, id, JointField.bodyIdA) !== defs[i].bodyIdA)
                throw new Error("joint allocation subject lost its definition or identity");
            writeJointFloat(state, id, J_EVENT, 1);
            collectJointEvents(state);
            if (
                eventCount(state, EventKind.Joint) !== 1 ||
                kernel(state.ecsState).eventWord(state.worldId, EventKind.Joint, 0, 0) !== id + 1 ||
                state.jointEventUserData[0] !== i
            )
                throw new Error("joint allocation subject lost its internal event");
            destroyJointInternal(state, id, false);
        }
        for (let i = 0; i < proxies.length; ++i) bufferMove(state.broadPhase, proxies[i]);
        updateBroadPhasePairs(state);
        if (contactCount(state) !== 0)
            throw new Error("joint allocation subject lost its pair filtering");
        control?.();
    };
}
