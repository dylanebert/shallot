import {
    bodyStructWGSL,
    constraintStructWGSL,
    MAX_COLORS,
    DEBUG_OFFSET,
    NUM_PAIR_TYPES,
    SS_PAIR_TYPE_BASE,
    SS_CONTACT_COUNT,
    SS_CONTACT_OVERFLOW,
    MAX_CONTACTS,
    CONTACT_STRIDE,
} from "./solver.wgsl";

export const COMPACT_STRIDE = 7;
export const COMPACT_BYTES = COMPACT_STRIDE * 4;

export const BODY_STRIDE = 52;
export const BODY_BYTES = BODY_STRIDE * 4;
export const BODY_QUAT_OFFSET = 20;

export const prepareIndirectWGSL = /* wgsl */ `
const MAX_COLORS: u32 = ${MAX_COLORS}u;
const DEBUG_OFFSET: u32 = ${DEBUG_OFFSET}u;

@group(0) @binding(0) var<storage> solverState: array<u32>;
@group(0) @binding(1) var<storage, read_write> indirectDispatch: array<u32>;

@compute @workgroup_size(1)
fn main() {
    for (var c = 0u; c < MAX_COLORS; c++) {
        let count = solverState[DEBUG_OFFSET + c];
        let wg = (count + 63u) / 64u;
        indirectDispatch[c * 3u] = wg;
        indirectDispatch[c * 3u + 1u] = 1u;
        indirectDispatch[c * 3u + 2u] = 1u;
    }
    let constraintCount = solverState[0u];
    let constraintWG = (constraintCount + 63u) / 64u;
    indirectDispatch[MAX_COLORS * 3u] = constraintWG;
    indirectDispatch[MAX_COLORS * 3u + 1u] = 1u;
    indirectDispatch[MAX_COLORS * 3u + 2u] = 1u;
    var totalPairs = 0u;
    for (var t = 0u; t < ${NUM_PAIR_TYPES}u; t++) {
        let typeCount = solverState[${SS_PAIR_TYPE_BASE}u + t];
        totalPairs += typeCount;
        let typeWG = (typeCount + 63u) / 64u;
        let off = (MAX_COLORS + 1u + t) * 3u;
        indirectDispatch[off] = typeWG;
        indirectDispatch[off + 1u] = 1u;
        indirectDispatch[off + 2u] = 1u;
    }
}
`;

export const compactWGSL = /* wgsl */ `

${bodyStructWGSL}

@group(0) @binding(0) var<storage> bodies: array<Body>;
@group(0) @binding(1) var<storage, read_write> compact: array<f32>;
@group(0) @binding(2) var<uniform> bodyCount: u32;

@compute @workgroup_size(64)
fn readback(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= bodyCount) { return; }
    let o = i * 7u;
    let pos = bodies[i].pos;
    let quat = bodies[i].quat;
    compact[o]     = pos.x;
    compact[o + 1u] = pos.y;
    compact[o + 2u] = pos.z;
    compact[o + 3u] = quat.x;
    compact[o + 4u] = quat.y;
    compact[o + 5u] = quat.z;
    compact[o + 6u] = quat.w;
}
`;

export const emitContactsWGSL = /* wgsl */ `

${bodyStructWGSL}
${constraintStructWGSL}

const SS_CONSTRAINT_COUNT: u32 = 0u;
const SS_CONTACT_COUNT: u32 = ${SS_CONTACT_COUNT}u;
const SS_CONTACT_OVERFLOW: u32 = ${SS_CONTACT_OVERFLOW}u;
const MAX_CONTACTS: u32 = ${MAX_CONTACTS}u;
const CONTACT_STRIDE: u32 = ${CONTACT_STRIDE}u;
const CONTACT_IMPULSE_THRESHOLD: f32 = 0.01;

@group(0) @binding(0) var<storage> bodies: array<Body>;
@group(0) @binding(1) var<storage> constraints: array<GPUConstraint>;
@group(0) @binding(2) var<storage, read_write> solverState: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> contacts: array<u32>;

@compute @workgroup_size(64)
fn emitContacts(@builtin(global_invocation_id) gid: vec3u) {
    let ci = gid.x;
    if (ci >= atomicLoad(&solverState[SS_CONSTRAINT_COUNT])) { return; }

    let c = constraints[ci];
    if (c.bilateral != CONSTRAINT_CONTACT && c.bilateral != CONSTRAINT_KINEMATIC) { return; }
    if (c.isNew == 0u) { return; }
    // lambda_n is negative in AVBD (clamped to [-1e30, 0]); negate for physical impulse
    let impulse = -c.lambda_n;
    if (impulse <= CONTACT_IMPULSE_THRESHOLD) { return; }

    let slot = atomicAdd(&solverState[SS_CONTACT_COUNT], 1u);
    if (slot >= MAX_CONTACTS) {
        atomicAdd(&solverState[SS_CONTACT_OVERFLOW], 1u);
        return;
    }
    let pos = bodies[c.bodyA].pos + c.rAW;
    let base = slot * CONTACT_STRIDE;
    contacts[base + 0u] = c.bodyA;
    contacts[base + 1u] = bitcast<u32>(c.bodyB);
    contacts[base + 2u] = bitcast<u32>(pos.x);
    contacts[base + 3u] = bitcast<u32>(pos.y);
    contacts[base + 4u] = bitcast<u32>(pos.z);
    contacts[base + 5u] = bitcast<u32>(c.normal.x);
    contacts[base + 6u] = bitcast<u32>(c.normal.y);
    contacts[base + 7u] = bitcast<u32>(c.normal.z);
    contacts[base + 8u] = bitcast<u32>(impulse);
}
`;

export const packWGSL = /* wgsl */ `

${bodyStructWGSL}

const SHAPE_BOX: f32 = 0.0;
const SHAPE_SPHERE: f32 = 1.0;
const SHAPE_CAPSULE: f32 = 2.0;
const SHAPE_HULL: f32 = 3.0;
struct PackParams {
    bodyCount: u32,
    section: u32,
    offset: u32,
}

@group(0) @binding(0) var<storage> sizes: array<f32>;
@group(0) @binding(1) var<storage> shapes: array<u32>;
@group(0) @binding(2) var<storage> bodyProps: array<f32>;
@group(0) @binding(3) var<storage> eids: array<u32>;
@group(0) @binding(4) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(5) var<uniform> packParams: PackParams;
@group(0) @binding(6) var<storage> transform: array<f32>;
@group(0) @binding(7) var<storage> hullIds: array<u32>;

@compute @workgroup_size(64)
fn packBodies(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= packParams.bodyCount) { return; }

    let eid = eids[i];
    let S = packParams.section;

    let sizeOff = eid * 4u;
    let hx = sizes[sizeOff] / 2.0;
    let hy = sizes[sizeOff + 1u] / 2.0;
    let hz = sizes[sizeOff + 2u] / 2.0;
    let shapeRadius = hx;
    let boundingRadius = length(vec3f(hx, hy, hz));

    let shapeByte = (shapes[eid / 4u] >> ((eid % 4u) * 8u)) & 0xFFu;
    let isSphere = shapeByte == 1u;
    let isCapsule = shapeByte == 2u;
    let isHull = shapeByte == 255u;
    let isBox = !isSphere && !isCapsule && !isHull;

    let propsOff = eid * 4u;
    let m = bodyProps[propsOff];
    let fric = bodyProps[propsOff + 1u];
    let grav = bodyProps[propsOff + 2u];
    let group = u32(bodyProps[propsOff + 3u]);

    var momentX: f32;
    var momentY: f32;
    var momentZ: f32;
    if (isBox || isHull) {
        momentX = (m / 3.0) * (hy * hy + hz * hz);
        momentY = (m / 3.0) * (hx * hx + hz * hz);
        momentZ = (m / 3.0) * (hx * hx + hy * hy);
    } else if (isCapsule) {
        let r = shapeRadius;
        let h = 2.0 * hy;
        let cylVol = h;
        let sphVol = 4.0 * r / 3.0;
        let totalVol = cylVol + sphVol;
        let mCyl = m * cylVol / max(totalVol, 1e-12);
        let mHs = m * sphVol * 0.5 / max(totalVol, 1e-12);
        momentY = 0.5 * mCyl * r * r + 0.8 * mHs * r * r;
        momentX = mCyl * (3.0 * r * r + h * h) / 12.0
                + 2.0 * mHs * (0.4 * r * r + h * h / 4.0 + 0.375 * h * r);
        momentZ = momentX;
    } else {
        let r = shapeRadius;
        let I = (2.0 / 5.0) * m * r * r;
        momentX = I;
        momentY = I;
        momentZ = I;
    }

    var colliderType = SHAPE_BOX;
    if (isSphere) { colliderType = SHAPE_SPHERE; }
    else if (isCapsule) { colliderType = SHAPE_CAPSULE; }
    else if (isHull) { colliderType = SHAPE_HULL; }

    let radius = select(boundingRadius, shapeRadius, isSphere || isCapsule);
    let halfExtents = vec3f(hx, hy, hz);
    let hullId = select(0u, hullIds[i], isHull);

    if (i >= packParams.offset) {
        // New body — full initialization.
        var body: Body;
        body.pos = vec3f(transform[eid], transform[eid + S], transform[eid + S * 2u]);
        body.mass = m;
        body.vel = vec3f(0.0);
        body.momentX = momentX;
        body.angVel = vec3f(0.0);
        body.radius = radius;
        body.inertial = vec3f(0.0);
        body.friction = fric;
        body.initial = vec3f(0.0);
        body.hullId = hullId;
        body.quat = vec4f(
            transform[eid + S * 3u],
            transform[eid + S * 4u],
            transform[eid + S * 5u],
            transform[eid + S * 6u]
        );
        body.inertialQuat = vec4f(0.0, 0.0, 0.0, 1.0);
        body.initialQuat = vec4f(0.0, 0.0, 0.0, 1.0);
        body.prevVel = vec3f(0.0);
        body.momentY = momentY;
        body.prevAngVel = vec3f(0.0);
        body.momentZ = momentZ;
        body.cumAng = vec3f(0.0);
        body.gravity = grav;
        body.halfExtents = halfExtents;
        body.colliderType = colliderType;
        body.collisionGroup = group;
        body.moved = 0.0;
        body._pad50 = 0.0;
        body._pad51 = 0.0;
        bodies[i] = body;
        return;
    }

    // Existing body — refresh shape-derived fields without disturbing dynamic state.
    bodies[i].mass = m;
    bodies[i].momentX = momentX;
    bodies[i].radius = radius;
    bodies[i].friction = fric;
    bodies[i].hullId = hullId;
    bodies[i].momentY = momentY;
    bodies[i].momentZ = momentZ;
    bodies[i].gravity = grav;
    bodies[i].halfExtents = halfExtents;
    bodies[i].colliderType = colliderType;
    bodies[i].collisionGroup = group;
}
`;

export const rebuildWGSL = /* wgsl */ `

${constraintStructWGSL}

struct RebuildParams {
    prevConstraintCount: u32,
    hashCapacity: u32,
    _pad0: u32,
    _pad1: u32,
}

const HASH_EMPTY: u32 = 0xFFFFFFFFu;
const MAX_PROBE: u32 = 128u;
const PENALTY_MIN: f32 = 1.0;
const PENALTY_MAX: f32 = 1e10;
const FEATURE_KEY_NONE: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage, read_write> hashKeys: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> warmstarts: array<WarmstartEntry>;
@group(0) @binding(2) var<storage, read> prevConstraints: array<GPUConstraint>;
@group(0) @binding(3) var<uniform> rebuildParams: RebuildParams;

fn defaultWarmstart() -> WarmstartEntry {
    return WarmstartEntry(0.0, PENALTY_MIN, 0.0, PENALTY_MIN, 0.0, PENALTY_MIN, 0u, FEATURE_KEY_NONE, vec3f(0.0), 0.0, vec3f(0.0), 0.0);
}

fn hashKey(k: u32) -> u32 {
    var h = k;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return h;
}

fn hashInsert(key: u32) -> u32 {
    let hCap = rebuildParams.hashCapacity;
    let mask = hCap - 1u;
    var slot = hashKey(key) & mask;
    for (var p = 0u; p < MAX_PROBE; p++) {
        let idx = (slot + p) & mask;
        let old = atomicCompareExchangeWeak(&hashKeys[idx], HASH_EMPTY, key);
        if (old.exchanged || old.old_value == key) { return idx; }
    }
    return hCap;
}

@compute @workgroup_size(64)
fn clearHash(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= rebuildParams.hashCapacity) { return; }
    atomicStore(&hashKeys[idx], HASH_EMPTY);
    warmstarts[idx] = defaultWarmstart();
}

@compute @workgroup_size(64)
fn rebuildWarm(@builtin(global_invocation_id) gid: vec3u) {
    let ci = gid.x;
    if (ci >= rebuildParams.prevConstraintCount) { return; }

    let c = prevConstraints[ci];
    let slot = hashInsert(c.warmstartIdx);
    if (slot >= rebuildParams.hashCapacity) { return; }

    let ws = WarmstartEntry(
        c.lambda_n, c.penalty_n,
        c.lambda_t1, c.penalty_t1,
        c.lambda_t2, c.penalty_t2,
        c.stick, c.featureKey,
        c.rA, 0.0,
        c.rB, 0.0,
    );
    warmstarts[slot] = ws;
}

`;

export const syncTransformsWGSL = /* wgsl */ `

${bodyStructWGSL}

struct SyncParams {
    bodyCount: u32,
    section: u32,
}

@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(1) var<storage> eids: array<u32>;
@group(0) @binding(2) var<storage> transform: array<f32>;
@group(0) @binding(3) var<uniform> params: SyncParams;

@compute @workgroup_size(64)
fn syncTransforms(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.bodyCount) { return; }
    if (bodies[i].mass > 0.0) { return; }

    let eid = eids[i];
    let S = params.section;
    let newPos = vec3f(
        transform[eid],
        transform[eid + S],
        transform[eid + S * 2u]
    );
    let moveFlag = transform[eid + S * 7u];
    if (moveFlag > 0.5) {
        let diff = newPos - bodies[i].pos;
        bodies[i].moved = select(0.0, 1.0, dot(diff, diff) > 1e-10);
        bodies[i].vel = diff;
    } else {
        bodies[i].moved = 0.0;
        bodies[i].vel = vec3f(0.0);
    }
    bodies[i].pos = newPos;
    bodies[i].quat = vec4f(
        transform[eid + S * 3u],
        transform[eid + S * 4u],
        transform[eid + S * 5u],
        transform[eid + S * 6u]
    );
}
`;
