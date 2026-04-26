import { solverTypesWGSL, broadphaseBindingsWGSL } from "./solver.wgsl";

export const bvhTraversalWGSL = /* wgsl */ `
${solverTypesWGSL}
${broadphaseBindingsWGSL}

@group(1) @binding(0) var<storage, read_write> pairs: array<u32>;

fn isJointed(a: u32, b: u32) -> bool {
    for (var ji = 0u; ji < params.jointCount; ji++) {
        let j = joints[ji];
        if ((j.bodyA == a && j.bodyB == b) || (j.bodyA == b && j.bodyB == a)) {
            return true;
        }
    }
    return false;
}

fn aabbOverlap(minA: vec3f, maxA: vec3f, minB: vec3f, maxB: vec3f) -> bool {
    return minA.x <= maxB.x && maxA.x >= minB.x
        && minA.y <= maxB.y && maxA.y >= minB.y
        && minA.z <= maxB.z && maxA.z >= minB.z;
}

const PAIR_TYPE_LUT = array<u32, 16>(
    0u, 1u, 2u, 6u,
    0u, 3u, 4u, 7u,
    0u, 0u, 5u, 8u,
    0u, 0u, 0u, 9u,
);

fn emitPair(a: u32, b: u32) {
    let itA = u32(bodies[a].colliderType);
    let itB = u32(bodies[b].colliderType);
    let lo = min(itA, itB);
    let hi = max(itA, itB);
    let pairType = PAIR_TYPE_LUT[lo * 4u + hi];

    var first: u32;
    var second: u32;
    if (lo == hi) {
        first = min(a, b);
        second = max(a, b);
    } else if (itA > itB) {
        first = a;
        second = b;
    } else {
        first = b;
        second = a;
    }

    let maxPerType = params.capacity * params.constraintMul;
    let pi = atomicAdd(&solverState[SS_PAIR_TYPE_BASE + pairType], 1u);
    if (pi >= maxPerType) { return; }
    let base = pairType * maxPerType;
    pairs[(base + pi) * 2u] = first;
    pairs[(base + pi) * 2u + 1u] = second;
}

fn testBinaryChild(
    child: u32,
    myMin: vec3f, myMax: vec3f,
    idx: u32,
    stack: ptr<function, array<u32, 64>>,
    stackPtr: ptr<function, u32>,
) {
    if ((child & LEAF_FLAG) != 0u) {
        let leafIdx = child & ~LEAF_FLAG;
        let otherIdx = sortedBodyIds[leafIdx];
        if (otherIdx == idx) { return; }
        if (bodies[otherIdx].mass > 0.0 && idx >= otherIdx) { return; }
        let la = leafAABBs[otherIdx];
        if (aabbOverlap(myMin, myMax, vec3f(la.minX, la.minY, la.minZ), vec3f(la.maxX, la.maxY, la.maxZ))) {
            if (isJointed(idx, otherIdx)) { return; }
            let groupA = bodies[idx].collisionGroup;
            let groupB = bodies[otherIdx].collisionGroup;
            if (groupA != 0u && groupA == groupB) { return; }
            emitPair(idx, otherIdx);
        }
    } else {
        let node = treeNodes[child];
        let nodeMin = vec3f(node.minX, node.minY, node.minZ);
        let nodeMax = vec3f(node.maxX, node.maxY, node.maxZ);
        if (aabbOverlap(myMin, myMax, nodeMin, nodeMax)) {
            if (*stackPtr < 64u) {
                (*stack)[*stackPtr] = child;
                *stackPtr += 1u;
            } else {
                atomicAdd(&solverState[SS_STACK_OVERFLOW], 1u);
            }
        }
    }
}

@compute @workgroup_size(64)
fn broadphase(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.bodyCount) { return; }

    let idx = sortedBodyIds[gid.x];
    if (bodies[idx].mass <= 0.0) { return; }

    let la = leafAABBs[idx];
    let myMin = vec3f(la.minX, la.minY, la.minZ);
    let myMax = vec3f(la.maxX, la.maxY, la.maxZ);

    if (idx == 0u) {
        let root = treeNodes[0];
        atomicStore(&solverState[DEBUG_BROADPHASE + 0u], 0xBEEFu);
        atomicStore(&solverState[DEBUG_BROADPHASE + 1u], params.bodyCount);
        atomicStore(&solverState[DEBUG_BROADPHASE + 2u], bitcast<u32>(root.minX));
        atomicStore(&solverState[DEBUG_BROADPHASE + 3u], bitcast<u32>(root.minY));
        atomicStore(&solverState[DEBUG_BROADPHASE + 4u], bitcast<u32>(root.minZ));
        atomicStore(&solverState[DEBUG_BROADPHASE + 5u], bitcast<u32>(root.maxX));
        atomicStore(&solverState[DEBUG_BROADPHASE + 6u], bitcast<u32>(root.maxY));
        atomicStore(&solverState[DEBUG_BROADPHASE + 7u], bitcast<u32>(root.maxZ));
        atomicStore(&solverState[DEBUG_BROADPHASE + 8u], root.leftChild);
        atomicStore(&solverState[DEBUG_BROADPHASE + 9u], root.rightChild);
    }

    var stack: array<u32, 64>;
    var stackPtr: u32 = 0u;

    let root = treeNodes[0];
    testBinaryChild(root.leftChild, myMin, myMax, idx, &stack, &stackPtr);
    testBinaryChild(root.rightChild, myMin, myMax, idx, &stack, &stackPtr);

    while (stackPtr > 0u) {
        stackPtr -= 1u;
        let nodeIdx = stack[stackPtr];
        let node = treeNodes[nodeIdx];
        testBinaryChild(node.leftChild, myMin, myMax, idx, &stack, &stackPtr);
        testBinaryChild(node.rightChild, myMin, myMax, idx, &stack, &stackPtr);
    }
}
`;
