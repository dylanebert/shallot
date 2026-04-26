import { bodyStructWGSL } from "./solver.wgsl";

const characterTypesWGSL = /* wgsl */ `
struct CharacterData {
    maxSlope: f32,
    grounded: u32,
    moveX: f32,
    moveY: f32,
    moveZ: f32,
    mass: f32,
    _pad1: f32,
    _pad2: f32,
}

struct CharacterParams {
    count: u32,
}
`;

export const characterSweepWGSL = /* wgsl */ `
${bodyStructWGSL}
${characterTypesWGSL}

struct TreeNode {
    minX: f32,
    minY: f32,
    minZ: f32,
    leftChild: u32,
    maxX: f32,
    maxY: f32,
    maxZ: f32,
    rightChild: u32,
}

struct LeafAABB {
    minX: f32, minY: f32, minZ: f32, _pad0: u32,
    maxX: f32, maxY: f32, maxZ: f32, _pad1: u32,
}

const LEAF_FLAG: u32 = 0x80000000u;
const SHAPE_BOX: f32 = 0.0;
const SHAPE_SPHERE: f32 = 1.0;
const SHAPE_CAPSULE: f32 = 2.0;
const COLLISION_MARGIN: f32 = 0.01;
const MAX_SWEEP_ITERS: u32 = 4u;
const MAX_NEARBY: u32 = 32u;
const GATHER_EXPAND: f32 = 0.1;

@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(1) var<storage, read> treeNodes: array<TreeNode>;
@group(0) @binding(2) var<storage, read> sortedBodyIds: array<u32>;
@group(0) @binding(3) var<storage, read> leafAABBs: array<LeafAABB>;
@group(0) @binding(4) var<storage, read_write> charData: array<CharacterData>;
@group(0) @binding(5) var<storage, read> charIndices: array<u32>;
@group(0) @binding(6) var<uniform> charParams: CharacterParams;
@group(0) @binding(7) var<storage, read_write> charGroundIdx: array<u32>;

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
    let u = q.xyz;
    let t = 2.0 * cross(u, v);
    return v + q.w * t + cross(u, t);
}

fn quatConj(q: vec4f) -> vec4f {
    return vec4f(-q.x, -q.y, -q.z, q.w);
}

fn aabbOverlap(minA: vec3f, maxA: vec3f, minB: vec3f, maxB: vec3f) -> bool {
    return minA.x <= maxB.x && maxA.x >= minB.x
        && minA.y <= maxB.y && maxA.y >= minB.y
        && minA.z <= maxB.z && maxA.z >= minB.z;
}

struct Contact {
    normal: vec3f,
    depth: f32,
}

fn closestPointOnSegment(p: vec3f, a: vec3f, b: vec3f) -> vec3f {
    let ab = b - a;
    let ab2 = dot(ab, ab);
    if (ab2 < 1e-12) { return a; }
    let t = clamp(dot(p - a, ab) / ab2, 0.0, 1.0);
    return a + ab * t;
}

fn satAxis(charPos: vec3f, charH: vec3f, boxPos: vec3f, bx: vec3f, by: vec3f, bz: vec3f, boxH: vec3f, axis: vec3f) -> f32 {
    let projC = abs(axis.x) * charH.x + abs(axis.y) * charH.y + abs(axis.z) * charH.z;
    let projB = abs(dot(bx, axis)) * boxH.x + abs(dot(by, axis)) * boxH.y + abs(dot(bz, axis)) * boxH.z;
    let dist = abs(dot(boxPos - charPos, axis));
    return projC + projB - dist;
}

fn aabbVsOBB(charPos: vec3f, charH: vec3f, other: Body) -> Contact {
    let d = other.pos - charPos;
    let bx = quatRotate(other.quat, vec3f(1.0, 0.0, 0.0));
    let by = quatRotate(other.quat, vec3f(0.0, 1.0, 0.0));
    let bz = quatRotate(other.quat, vec3f(0.0, 0.0, 1.0));
    let bh = other.halfExtents;

    var bestDepth = 1e30;
    var bestNormal = vec3f(0.0);

    let axes = array<vec3f, 6>(
        vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0),
        bx, by, bz,
    );

    for (var i = 0u; i < 6u; i++) {
        let axis = axes[i];
        let overlap = satAxis(charPos, charH, other.pos, bx, by, bz, bh, axis);
        if (overlap <= 0.0) { return Contact(vec3f(0.0), 0.0); }
        if (overlap < bestDepth) {
            bestDepth = overlap;
            bestNormal = select(-axis, axis, dot(d, axis) < 0.0);
        }
    }

    return Contact(bestNormal, bestDepth);
}

fn aabbVsSphere(charPos: vec3f, charH: vec3f, other: Body) -> Contact {
    let sphR = other.halfExtents.x;
    let d = other.pos - charPos;
    let closest = clamp(d, -charH, charH);
    let diff = d - closest;
    let dist2 = dot(diff, diff);

    if (dist2 > (sphR + COLLISION_MARGIN) * (sphR + COLLISION_MARGIN) && dist2 > 1e-16) {
        return Contact(vec3f(0.0), 0.0);
    }

    let absD = abs(d);
    let inside = absD.x <= charH.x && absD.y <= charH.y && absD.z <= charH.z;

    if (!inside && dist2 > 1e-16) {
        let dist = sqrt(dist2);
        let gap = dist - sphR;
        if (gap > COLLISION_MARGIN) { return Contact(vec3f(0.0), 0.0); }
        let normal = -diff / dist;
        return Contact(normal, sphR - dist);
    }

    let face = charH - absD;
    var minAxis = 0u;
    var minVal = face.x;
    if (face.y < minVal) { minAxis = 1u; minVal = face.y; }
    if (face.z < minVal) { minAxis = 2u; minVal = face.z; }
    var normal = vec3f(0.0);
    if (minAxis == 0u) {
        normal.x = select(1.0, -1.0, d.x >= 0.0);
    } else if (minAxis == 1u) {
        normal.y = select(1.0, -1.0, d.y >= 0.0);
    } else {
        normal.z = select(1.0, -1.0, d.z >= 0.0);
    }
    return Contact(normal, minVal + sphR);
}

fn aabbVsCapsule(charPos: vec3f, charH: vec3f, other: Body) -> Contact {
    let capAxis = quatRotate(other.quat, vec3f(0.0, other.halfExtents.y, 0.0));
    let capR = other.halfExtents.x;
    let epA = other.pos + capAxis;
    let epB = other.pos - capAxis;

    let closest = closestPointOnSegment(charPos, epA, epB);
    let d = closest - charPos;
    let clamped = clamp(d, -charH, charH);
    let diff = d - clamped;
    let dist2 = dot(diff, diff);

    let absD = abs(d);
    let inside = absD.x <= charH.x && absD.y <= charH.y && absD.z <= charH.z;

    if (!inside && dist2 > 1e-16) {
        let dist = sqrt(dist2);
        if (dist - capR > COLLISION_MARGIN) { return Contact(vec3f(0.0), 0.0); }
        let normal = -diff / dist;
        return Contact(normal, capR - dist);
    }

    let face = charH - absD;
    var minAxis = 0u;
    var minVal = face.x;
    if (face.y < minVal) { minAxis = 1u; minVal = face.y; }
    if (face.z < minVal) { minAxis = 2u; minVal = face.z; }
    var normal = vec3f(0.0);
    if (minAxis == 0u) {
        normal.x = select(1.0, -1.0, d.x >= 0.0);
    } else if (minAxis == 1u) {
        normal.y = select(1.0, -1.0, d.y >= 0.0);
    } else {
        normal.z = select(1.0, -1.0, d.z >= 0.0);
    }
    return Contact(normal, minVal + capR);
}


fn testBody(charPos: vec3f, charH: vec3f, other: Body) -> Contact {
    if (other.colliderType == SHAPE_BOX) {
        return aabbVsOBB(charPos, charH, other);
    }
    if (other.colliderType == SHAPE_SPHERE) {
        return aabbVsSphere(charPos, charH, other);
    }
    if (other.colliderType == SHAPE_CAPSULE) {
        return aabbVsCapsule(charPos, charH, other);
    }
    return Contact(vec3f(0.0), 0.0);
}

@compute @workgroup_size(64)
fn characterSweep(@builtin(global_invocation_id) gid: vec3u) {
    let charIdx = gid.x;
    if (charIdx >= charParams.count) { return; }

    let bodyIdx = charIndices[charIdx];
    let body = bodies[bodyIdx];
    let cd = charData[charIdx];
    var pos = body.pos;
    let charH = body.halfExtents;
    let maxSlopeCos = cd.maxSlope;

    var nearby: array<u32, 32>;
    var nearbyCount: u32 = 0u;

    let gatherMargin = vec3f(COLLISION_MARGIN + GATHER_EXPAND);
    let gatherMin = pos - charH - gatherMargin;
    let gatherMax = pos + charH + gatherMargin;

    var stack: array<u32, 64>;
    var stackPtr: u32 = 0u;
    let root = treeNodes[0];

    for (var side = 0u; side < 2u; side++) {
        let child = select(root.rightChild, root.leftChild, side == 0u);
        if ((child & LEAF_FLAG) != 0u) {
            let oi = sortedBodyIds[child & ~LEAF_FLAG];
            if (oi != bodyIdx) {
                let la = leafAABBs[oi];
                if (aabbOverlap(gatherMin, gatherMax, vec3f(la.minX, la.minY, la.minZ), vec3f(la.maxX, la.maxY, la.maxZ))) {
                    if (nearbyCount < MAX_NEARBY) { nearby[nearbyCount] = oi; nearbyCount += 1u; }
                }
            }
        } else {
            let n = treeNodes[child];
            if (aabbOverlap(gatherMin, gatherMax, vec3f(n.minX, n.minY, n.minZ), vec3f(n.maxX, n.maxY, n.maxZ))) {
                stack[stackPtr] = child;
                stackPtr += 1u;
            }
        }
    }

    while (stackPtr > 0u) {
        stackPtr -= 1u;
        let node = treeNodes[stack[stackPtr]];

        for (var side = 0u; side < 2u; side++) {
            let child = select(node.rightChild, node.leftChild, side == 0u);
            if ((child & LEAF_FLAG) != 0u) {
                let oi = sortedBodyIds[child & ~LEAF_FLAG];
                if (oi != bodyIdx) {
                    let la = leafAABBs[oi];
                    if (aabbOverlap(gatherMin, gatherMax, vec3f(la.minX, la.minY, la.minZ), vec3f(la.maxX, la.maxY, la.maxZ))) {
                        if (nearbyCount < MAX_NEARBY) { nearby[nearbyCount] = oi; nearbyCount += 1u; }
                    }
                }
            } else {
                let n = treeNodes[child];
                if (aabbOverlap(gatherMin, gatherMax, vec3f(n.minX, n.minY, n.minZ), vec3f(n.maxX, n.maxY, n.maxZ))) {
                    if (stackPtr < 64u) { stack[stackPtr] = child; stackPtr += 1u; }
                }
            }
        }
    }

    var grounded = false;
    var groundBodyIdx = 0xFFFFFFFFu;

    for (var iter = 0u; iter < MAX_SWEEP_ITERS; iter++) {
        var bestNormal = vec3f(0.0);
        var bestDepth: f32 = 0.0;

        for (var ni = 0u; ni < nearbyCount; ni++) {
            let oi = nearby[ni];
            let c = testBody(pos, charH, bodies[oi]);
            if (c.depth > bestDepth) { bestDepth = c.depth; bestNormal = c.normal; }
            if (c.depth > 0.0 && c.normal.y > maxSlopeCos) { grounded = true; groundBodyIdx = oi; }
        }

        if (bestDepth <= 0.0) { break; }
        pos += bestNormal * bestDepth;
    }

    bodies[bodyIdx].pos = pos;
    charData[charIdx].grounded = select(0u, 1u, grounded);
    charGroundIdx[charIdx] = groundBodyIdx;
}
`;

export const characterApplyWGSL = /* wgsl */ `
${bodyStructWGSL}
${characterTypesWGSL}

@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(4) var<storage, read_write> charData: array<CharacterData>;
@group(0) @binding(5) var<storage, read> charIndices: array<u32>;
@group(0) @binding(6) var<uniform> charParams: CharacterParams;
@group(0) @binding(7) var<storage, read_write> charGroundIdx: array<u32>;

@compute @workgroup_size(64)
fn characterApply(@builtin(global_invocation_id) gid: vec3u) {
    let charIdx = gid.x;
    if (charIdx >= charParams.count) { return; }
    let bodyIdx = charIndices[charIdx];
    let cd = charData[charIdx];
    var charMove = vec3f(cd.moveX, cd.moveY, cd.moveZ);
    let groundIdx = charGroundIdx[charIdx];
    if (groundIdx != 0xFFFFFFFFu && bodies[groundIdx].mass <= 0.0) {
        charMove += bodies[groundIdx].vel;
    }
    bodies[bodyIdx].pos += charMove;
    bodies[bodyIdx].vel = charMove;
    if (dot(charMove, charMove) > 1e-10) {
        bodies[bodyIdx].moved = 1.0;
    }
}
`;
