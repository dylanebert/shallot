import {
    JOINT_BYTES,
    MAX_COLORS,
    MAX_DEGREE,
    SS_CONSTRAINT_COUNT,
    SS_CONSTRAINT_OVERFLOW,
    SS_USED_COLORS,
    SS_HASH_OVERFLOW,
    SS_STACK_OVERFLOW,
    SS_ADJ_OVERFLOW,
    SS_HASH_OCCUPANCY,
    maxConstraints,
    hashCapacity,
} from "./solver.wgsl";

export interface RemoveOp {
    removedIdx: number;
    lastIdx: number;
}

export interface ChangeResult {
    removeOps: RemoveOp[];
    addEids: number[];
}

const changeScratch = new Map<number, boolean>();
const removeIndicesScratch: number[] = [];

export function resolveChanges(
    pendingChanges: Int32Array,
    pendingChangeCount: number,
    bodyEids: number[],
    cap: number,
): ChangeResult {
    const finalState = changeScratch;
    finalState.clear();
    for (let i = 0; i < pendingChangeCount; i++) {
        const v = pendingChanges[i];
        if (v > 0) finalState.set(v - 1, true);
        else finalState.set(-v - 1, false);
    }

    const removeIndices = removeIndicesScratch;
    removeIndices.length = 0;
    for (const [eid, add] of finalState) {
        if (!add || bodyEids.indexOf(eid) >= 0) {
            const idx = bodyEids.indexOf(eid);
            if (idx >= 0) removeIndices.push(idx);
        }
    }

    const removeOps: RemoveOp[] = [];
    if (removeIndices.length > 0) {
        removeIndices.sort((a, b) => b - a);
        for (let i = 0; i < removeIndices.length; i++) {
            const removedIdx = removeIndices[i];
            const lastIdx = bodyEids.length - 1;
            if (removedIdx !== lastIdx) {
                bodyEids[removedIdx] = bodyEids[lastIdx];
                removeOps.push({ removedIdx, lastIdx });
            }
            bodyEids.length--;
        }
    }

    const addEids: number[] = [];
    for (const [eid, add] of finalState) {
        if (!add) continue;
        if (bodyEids.indexOf(eid) >= 0) continue;
        if (bodyEids.length >= cap) break;
        bodyEids.push(eid);
        addEids.push(eid);
    }

    return { removeOps, addEids };
}

export interface JointInput {
    anchorAX: number;
    anchorAY: number;
    anchorAZ: number;
    bodyA: number;
    anchorBX: number;
    anchorBY: number;
    anchorBZ: number;
    bodyB: number;
    stiffness: number;
    fracture: number;
}

export interface SpringJointInput extends JointInput {
    restLength: number;
}

export function packJointData(
    ballJoints: JointInput[],
    springJoints: SpringJointInput[],
    eidToIndex: Map<number, number>,
): Float32Array {
    const floatsPerJoint = JOINT_BYTES / 4;
    const count = ballJoints.length + springJoints.length;
    const jointData = new Float32Array(count * floatsPerJoint);
    const jointU32 = new Uint32Array(jointData.buffer);
    let i = 0;

    for (const j of ballJoints) {
        const off = i * floatsPerJoint;
        jointData[off + 0] = j.anchorAX;
        jointData[off + 1] = j.anchorAY;
        jointData[off + 2] = j.anchorAZ;
        jointU32[off + 3] = eidToIndex.get(j.bodyA) ?? 0;
        jointData[off + 4] = j.anchorBX;
        jointData[off + 5] = j.anchorBY;
        jointData[off + 6] = j.anchorBZ;
        jointU32[off + 7] = eidToIndex.get(j.bodyB) ?? 0;
        jointU32[off + 8] = 0;
        jointData[off + 10] = j.stiffness;
        jointData[off + 16] = j.fracture;
        i++;
    }

    for (const j of springJoints) {
        const off = i * floatsPerJoint;
        jointData[off + 0] = j.anchorAX;
        jointData[off + 1] = j.anchorAY;
        jointData[off + 2] = j.anchorAZ;
        jointU32[off + 3] = eidToIndex.get(j.bodyA) ?? 0;
        jointData[off + 4] = j.anchorBX;
        jointData[off + 5] = j.anchorBY;
        jointData[off + 6] = j.anchorBZ;
        jointU32[off + 7] = eidToIndex.get(j.bodyB) ?? 0;
        jointU32[off + 8] = 1;
        jointData[off + 9] = j.restLength;
        jointData[off + 10] = j.stiffness;
        jointData[off + 16] = j.fracture;
        i++;
    }

    return jointData;
}

export function checkOverflows(u32: Uint32Array, tick: number, bodyCount: number): void {
    const ov = u32[SS_CONSTRAINT_OVERFLOW];
    if (ov > 0)
        console.warn(
            `[phys] tick=${tick} CONSTRAINT OVERFLOW: ${ov}, count=${u32[SS_CONSTRAINT_COUNT]}, max=${maxConstraints()}, bodies=${bodyCount}`,
        );
    const usedColors = u32[SS_USED_COLORS];
    if (usedColors > MAX_COLORS)
        console.warn(
            `[phys] tick=${tick} COLOR OVERFLOW: scene needs ${usedColors} colors, max=${MAX_COLORS}`,
        );
    const hashOv = u32[SS_HASH_OVERFLOW];
    if (hashOv > 0)
        console.warn(
            `[phys] tick=${tick} HASH OVERFLOW: ${hashOv} inserts failed, capacity=${hashCapacity()}`,
        );
    const stackOv = u32[SS_STACK_OVERFLOW];
    if (stackOv > 0)
        console.warn(`[phys] tick=${tick} STACK OVERFLOW: ${stackOv} BVH traversals hit limit`);
    const adjOv = u32[SS_ADJ_OVERFLOW];
    if (adjOv > 0)
        console.warn(
            `[phys] tick=${tick} ADJACENCY OVERFLOW: ${adjOv} edges dropped (MAX_DEGREE=${MAX_DEGREE})`,
        );
    const occupancy = u32[SS_HASH_OCCUPANCY];
    const hCap = hashCapacity();
    if (occupancy > hCap * 0.75)
        console.warn(
            `[phys] tick=${tick} HASH OCCUPANCY: ${occupancy}/${hCap} (${((occupancy / hCap) * 100).toFixed(1)}%)`,
        );
}
