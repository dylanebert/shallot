import { capacity } from "../../engine";

export const MAX_PAIR_CONTACTS = 4;
export const JOINT_BYTES = 80;
export function maxConstraints() {
    return capacity() * 16;
}
export function hashCapacity() {
    return maxConstraints() * 2;
}
export const CONSTRAINT_BYTES = 176;
export const WARMSTART_BYTES = 64;
export const MAX_DEGREE = 32;
export const MAX_COLORS = 12;
export const SS_CONSTRAINT_COUNT = 0;
export const SS_WARMSTART_HITS = 3;
export const SS_CONSTRAINT_OVERFLOW = 4;
export const SS_STACK_OVERFLOW = 5;
export const SS_HASH_OVERFLOW = 6;
export const SS_USED_COLORS = 7;
export const DEBUG_OFFSET = 8;
export const NAN_COUNT_OFFSET = DEBUG_OFFSET + MAX_COLORS;
export const DEBUG_BROADPHASE = NAN_COUNT_OFFSET + 1;
export const DEBUG_BROADPHASE_SLOTS = 10;
export const SS_PENALTY_SATURATED = DEBUG_BROADPHASE + DEBUG_BROADPHASE_SLOTS;
export const SS_WARMSTART_NAN = SS_PENALTY_SATURATED + 1;
export const SS_HASH_COLLISIONS = SS_WARMSTART_NAN + 1;
export const SS_FEATURE_MISMATCH = SS_HASH_COLLISIONS + 1;
export const SS_WARMSTART_LOADED = SS_FEATURE_MISMATCH + 1;
export const SS_PAIR_COUNT = SS_WARMSTART_LOADED + 1;
export const SS_ADJ_OVERFLOW = SS_PAIR_COUNT + 1;
export const SS_UNCOLORED = SS_ADJ_OVERFLOW + 1;
export const SS_HASH_OCCUPANCY = SS_UNCOLORED + 1;
export const SS_CONTACT_COUNT = SS_HASH_OCCUPANCY + 1;
export const SS_CONTACT_OVERFLOW = SS_CONTACT_COUNT + 1;
export const NUM_PAIR_TYPES = 10;
export const SS_PAIR_TYPE_BASE = SS_CONTACT_OVERFLOW + 1;
export const COUNTERS_SIZE = (SS_PAIR_TYPE_BASE + NUM_PAIR_TYPES) * 4;

export const MAX_CONTACTS = 128;
export const CONTACT_STRIDE = 9;
const CONTACT_BYTES = CONTACT_STRIDE * 4;
export const CONTACTS_BYTES = MAX_CONTACTS * CONTACT_BYTES;

export const ADJ_STRIDE = 1 + MAX_DEGREE;
export const COLORING_ROUNDS = 16;

export const CSR_DATA_OFFSET = 0;
export function csrOffsetsOffset() {
    return 2 * maxConstraints();
}
export function csrHeadsOffset() {
    return csrOffsetsOffset() + capacity() + 1;
}
export function adjOffset() {
    return csrHeadsOffset() + capacity();
}
export function colorGraphSize() {
    return (adjOffset() + capacity() * ADJ_STRIDE) * 4;
}

export function sortedOffset() {
    return capacity();
}
export function colorMetaOffset() {
    return capacity() * 2;
}
export function colorStateSize() {
    return (capacity() * 2 + MAX_COLORS * 2 + 1) * 4;
}

export const HASH_REGION_OFFSET = Math.ceil(COUNTERS_SIZE / 256) * 256;
export function solverStateSize() {
    return HASH_REGION_OFFSET + hashCapacity() * 4;
}
export function colorDataSize() {
    return colorGraphSize() + colorStateSize();
}
export function colorDataBase() {
    return solverStateSize() / 4;
}

export const bodyStructWGSL = `struct Body {
    pos: vec3f,
    mass: f32,
    vel: vec3f,
    momentX: f32,
    angVel: vec3f,
    radius: f32,
    inertial: vec3f,
    friction: f32,
    initial: vec3f,
    hullId: u32,
    quat: vec4f,
    inertialQuat: vec4f,
    initialQuat: vec4f,
    prevVel: vec3f,
    momentY: f32,
    prevAngVel: vec3f,
    momentZ: f32,
    cumAng: vec3f,
    gravity: f32,
    halfExtents: vec3f,
    colliderType: f32,
    collisionGroup: u32,
    moved: f32,
    _pad50: f32,
    _pad51: f32,
}`;

export const constraintStructWGSL = /* wgsl */ `
struct GPUConstraint {
    bodyA: u32,
    bodyB: i32,
    featureKey: u32,
    stick: u32,
    normal: vec3f,
    C_init_n: f32,
    tangent1: vec3f,
    C_init_t1: f32,
    tangent2: vec3f,
    C_init_t2: f32,
    rA: vec3f,
    lambda_n: f32,
    rB: vec3f,
    penalty_n: f32,
    rAW: vec3f,
    friction: f32,
    lambda_t1: f32,
    penalty_t1: f32,
    lambda_t2: f32,
    penalty_t2: f32,
    isNew: u32,
    warmstartIdx: u32,
    bilateral: u32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    fmin_n: f32,
    fmax_n: f32,
    stiffness: f32,
    rBW_x: f32,
    rBW_y: f32,
    rBW_z: f32,
}

const CONSTRAINT_CONTACT = 0u;
const CONSTRAINT_BALL = 1u;
const CONSTRAINT_SPRING = 2u;
const CONSTRAINT_KINEMATIC = 3u;

struct WarmstartEntry {
    lambda_n: f32,
    penalty_n: f32,
    lambda_t1: f32,
    penalty_t1: f32,
    lambda_t2: f32,
    penalty_t2: f32,
    stick: u32,
    featureKey: u32,
    rA: vec3f,
    _pad0: f32,
    rB: vec3f,
    _pad1: f32,
}
`;

export const solverTypesWGSL = /* wgsl */ `

${bodyStructWGSL}
${constraintStructWGSL}

struct Params {
    dt: f32,
    gravity: f32,
    iterations: u32,
    alpha: f32,
    betaLin: f32,
    gamma: f32,
    bodyCount: u32,
    jointCount: u32,
    capacity: u32,
    constraintMul: u32,
    hashMul: u32,
    betaAng: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

const PENALTY_MIN: f32 = 1.0;
const PENALTY_MAX: f32 = 1e10;
const COLLISION_MARGIN: f32 = 0.01;
const STICK_THRESH: f32 = 1e-5;
const MAX_PAIR_CONTACTS: u32 = ${MAX_PAIR_CONTACTS}u;
const HASH_EMPTY: u32 = 0xFFFFFFFFu;
const MAX_PROBE: u32 = 128u;
const SHAPE_BOX: f32 = 0.0;
const SHAPE_SPHERE: f32 = 1.0;
const SHAPE_CAPSULE: f32 = 2.0;

const FEATURE_KEY_NONE: u32 = 0xFFFFFFFFu;

const SS_CONSTRAINT_COUNT: u32 = 0u;
const SS_WARMSTART_HITS: u32 = 3u;
const SS_CONSTRAINT_OVERFLOW: u32 = 4u;
const SS_STACK_OVERFLOW: u32 = 5u;
const DEBUG_BROADPHASE: u32 = ${DEBUG_BROADPHASE}u;
const SS_WARMSTART_NAN: u32 = ${SS_WARMSTART_NAN}u;
const SS_WARMSTART_LOADED: u32 = ${SS_WARMSTART_LOADED}u;
const NUM_PAIR_TYPES: u32 = ${NUM_PAIR_TYPES}u;
const SS_PAIR_TYPE_BASE: u32 = ${SS_PAIR_TYPE_BASE}u;
const SS_CONTACT_COUNT: u32 = ${SS_CONTACT_COUNT}u;
const SS_CONTACT_OVERFLOW: u32 = ${SS_CONTACT_OVERFLOW}u;
const MAX_CONTACTS: u32 = ${MAX_CONTACTS}u;
const CONTACT_STRIDE: u32 = ${CONTACT_STRIDE}u;
const HASH_BASE: u32 = ${HASH_REGION_OFFSET / 4}u;
`;

const jointStructWGSL = /* wgsl */ `
struct Joint {
    localAnchorA: vec3f,
    bodyA: u32,
    localAnchorB: vec3f,
    bodyB: u32,
    jointType: u32,
    restLength: f32,
    stiffness: f32,
    targetSpeed: f32,
    axis: vec3f,
    maxTorque: f32,
    fracture: f32,
    broken: u32,
    _pad0: f32,
    _pad1: f32,
}
`;

const bvhTypesWGSL = /* wgsl */ `
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

const LEAF_FLAG: u32 = 0x80000000u;

struct LeafAABB {
    minX: f32, minY: f32, minZ: f32, _pad0: u32,
    maxX: f32, maxY: f32, maxZ: f32, _pad1: u32,
}
`;

export const narrowphaseBindingsWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> constraints: array<GPUConstraint>;
@group(0) @binding(3) var<storage, read_write> warmstarts: array<WarmstartEntry>;
@group(0) @binding(5) var<storage, read_write> solverState: array<atomic<u32>>;
`;

export const broadphaseBindingsWGSL = /* wgsl */ `
${jointStructWGSL}
${bvhTypesWGSL}
@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> joints: array<Joint>;
@group(0) @binding(5) var<storage, read_write> solverState: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> treeNodes: array<TreeNode>;
@group(0) @binding(7) var<storage, read> sortedBodyIds: array<u32>;
@group(0) @binding(8) var<storage, read> leafAABBs: array<LeafAABB>;
`;

const allBindingsWGSL = /* wgsl */ `
${jointStructWGSL}
${bvhTypesWGSL}
@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> constraints: array<GPUConstraint>;
@group(0) @binding(3) var<storage, read_write> warmstarts: array<WarmstartEntry>;
@group(0) @binding(4) var<storage, read_write> joints: array<Joint>;
@group(0) @binding(5) var<storage, read_write> solverState: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> treeNodes: array<TreeNode>;
@group(0) @binding(7) var<storage, read> sortedBodyIds: array<u32>;
@group(0) @binding(8) var<storage, read> leafAABBs: array<LeafAABB>;
@group(0) @binding(9) var<storage, read> forces: array<f32>;
@group(0) @binding(10) var<storage, read_write> bodyCols: array<vec4f>;
`;

export const BODY_COL_COUNT = 5;
const BODY_COL_POS = 0;
const BODY_COL_INITIAL = 1;
const BODY_COL_CUMANG = 2;
const BODY_COL_QUAT = 3;
const BODY_COL_INITIAL_QUAT = 4;

const solverBaseWGSL = /* wgsl */ `
${solverTypesWGSL}
${allBindingsWGSL}
`;

export const solverCoreFnsWGSL = /* wgsl */ `

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
    let u = q.xyz;
    let t = 2.0 * cross(u, v);
    return v + q.w * t + cross(u, t);
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

fn hashLookup(key: u32) -> u32 {
    let hCap = params.capacity * params.hashMul;
    let mask = hCap - 1u;
    var slot = hashKey(key) & mask;
    for (var p = 0u; p < MAX_PROBE; p++) {
        let idx = (slot + p) & mask;
        let stored = atomicLoad(&solverState[HASH_BASE + idx]);
        if (stored == key) { return idx; }
        if (stored == HASH_EMPTY) { return hCap; }
    }
    return hCap;
}

fn tangentBasis(n: vec3f) -> array<vec3f, 2> {
    var t1: vec3f;
    if (abs(n.x) > abs(n.y)) {
        t1 = vec3f(-n.z, 0.0, n.x);
    } else {
        t1 = vec3f(0.0, n.z, -n.y);
    }
    t1 = normalize(t1);
    let t2 = cross(t1, n);
    return array<vec3f, 2>(t1, t2);
}

fn defaultWarmstart() -> WarmstartEntry {
    return WarmstartEntry(0.0, PENALTY_MIN, 0.0, PENALTY_MIN, 0.0, PENALTY_MIN, 0u, FEATURE_KEY_NONE, vec3f(0.0), 0.0, vec3f(0.0), 0.0);
}

fn isNanOrInf(v: f32) -> bool {
    return !(v == v) || abs(v) > 1e30;
}

fn applyWarmstart(ws: WarmstartEntry, stiffnessCap: f32) -> array<f32, 6> {
    let a = params.alpha;
    let g = params.gamma;
    return array<f32, 6>(
        (ws.lambda_n * a) * g,
        min(clamp(ws.penalty_n * g, PENALTY_MIN, PENALTY_MAX), stiffnessCap),
        (ws.lambda_t1 * a) * g,
        min(clamp(ws.penalty_t1 * g, PENALTY_MIN, PENALTY_MAX), stiffnessCap),
        (ws.lambda_t2 * a) * g,
        min(clamp(ws.penalty_t2 * g, PENALTY_MIN, PENALTY_MAX), stiffnessCap),
    );
}

fn pushConstraintWithWarmstart(
    bodyA: u32, bodyB: i32, featureKey: u32,
    normal: vec3f, C_init_n: f32,
    tangent1: vec3f, C_init_t1: f32,
    tangent2: vec3f, C_init_t2: f32,
    rA: vec3f, rB: vec3f,
    friction: f32,
    wsKey: u32, bilateral: u32,
    fmin_n: f32, fmax_n: f32, cStiffness: f32,
    ws: WarmstartEntry,
) {
    var warm = applyWarmstart(ws, cStiffness);
    if (bilateral == CONSTRAINT_KINEMATIC) {
        warm[0] = 0.0;
        warm[2] = 0.0;
        warm[4] = 0.0;
    }
    let isNew: u32 = select(0u, 1u, ws.featureKey == FEATURE_KEY_NONE);
    if (isNew == 0u) {
        atomicAdd(&solverState[SS_WARMSTART_HITS], 1u);
    }

    let ci = atomicAdd(&solverState[SS_CONSTRAINT_COUNT], 1u);
    if (ci >= params.capacity * params.constraintMul) { atomicAdd(&solverState[SS_CONSTRAINT_OVERFLOW], 1u); return; }
    constraints[ci] = GPUConstraint(
        bodyA, bodyB, featureKey, ws.stick,
        normal, C_init_n,
        tangent1, C_init_t1,
        tangent2, C_init_t2,
        rA, warm[0],
        rB, warm[1],
        vec3f(0.0), friction,
        warm[2], warm[3], warm[4], warm[5],
        isNew, wsKey, bilateral, 0.0,
        0.0, 0.0, fmin_n, fmax_n,
        cStiffness, 0.0, 0.0, 0.0,
    );
}
`;

const solverLookupFnsWGSL = /* wgsl */ `
fn quatMul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

fn hashInsert(key: u32) -> u32 {
    let hCap = params.capacity * params.hashMul;
    let mask = hCap - 1u;
    var slot = hashKey(key) & mask;
    for (var p = 0u; p < MAX_PROBE; p++) {
        let idx = (slot + p) & mask;
        let old = atomicCompareExchangeWeak(&solverState[HASH_BASE + idx], HASH_EMPTY, key);
        if (old.exchanged || old.old_value == key) {
            return idx;
        }
    }
    return hCap;
}

fn loadWarmstartHash(key: u32, featureKey: u32) -> WarmstartEntry {
    let hCap = params.capacity * params.hashMul;
    let idx = hashLookup(key);
    if (idx < hCap) {
        let ws = warmstarts[idx];
        if (ws.featureKey == featureKey && featureKey != FEATURE_KEY_NONE) {
            if (isNanOrInf(ws.lambda_n) || isNanOrInf(ws.penalty_n) ||
                isNanOrInf(ws.lambda_t1) || isNanOrInf(ws.penalty_t1) ||
                isNanOrInf(ws.lambda_t2) || isNanOrInf(ws.penalty_t2)) {
                atomicAdd(&solverState[SS_WARMSTART_NAN], 1u);
                return defaultWarmstart();
            }
            if (ws.penalty_n > PENALTY_MIN) {
                atomicAdd(&solverState[SS_WARMSTART_LOADED], 1u);
            }
            return ws;
        }
    }
    return defaultWarmstart();
}

fn pushConstraint(
    bodyA: u32, bodyB: i32, featureKey: u32,
    normal: vec3f, C_init_n: f32,
    tangent1: vec3f, C_init_t1: f32,
    tangent2: vec3f, C_init_t2: f32,
    rA: vec3f, rB: vec3f,
    friction: f32,
    wsKey: u32, bilateral: u32,
    fmin_n: f32, fmax_n: f32, cStiffness: f32,
) {
    let ws = loadWarmstartHash(wsKey, featureKey);
    pushConstraintWithWarmstart(bodyA, bodyB, featureKey, normal, C_init_n, tangent1, C_init_t1, tangent2, C_init_t2, rA, rB, friction, wsKey, bilateral, fmin_n, fmax_n, cStiffness, ws);
}
`;

const solverFunctionsWGSL = /* wgsl */ `
${solverCoreFnsWGSL}
${solverLookupFnsWGSL}
`;

export const sharedSolverWGSL = /* wgsl */ `
${solverBaseWGSL}
${solverFunctionsWGSL}
`;

export const packKeyWGSL = /* wgsl */ `
fn packKey(bodyA: u32, bodyB: u32, slot: u32) -> u32 {
    let lo = min(bodyA, bodyB);
    let hi = max(bodyA, bodyB);
    var h = lo * 0x9e3779b9u + hi;
    h ^= slot * 0x517cc1b7u;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return select(h, h ^ 1u, h == HASH_EMPTY);
}
`;

const coloringConstantsWGSL = /* wgsl */ `

const MAX_ANGVEL: f32 = 50.0;
const SOLVER_SHAPE_SPHERE: f32 = 1.0;
const SOLVER_SHAPE_CAPSULE: f32 = 2.0;
const SOLVER_SHAPE_HULL: f32 = 3.0;
const G_ZERO = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
const MAX_DEGREE: u32 = ${MAX_DEGREE}u;
const MAX_COLORS: u32 = ${MAX_COLORS}u;
const ADJ_STRIDE: u32 = ${ADJ_STRIDE}u;
const UNCOLORED: u32 = 0xFFFFFFFFu;
const KINEMATIC_COLOR: u32 = 0xFFFFFFFEu;

const SS_ITERATION: u32 = 1u;
const SS_CURRENT_COLOR: u32 = 2u;
const SS_HASH_OVERFLOW: u32 = 6u;
const SS_USED_COLORS: u32 = 7u;
const DEBUG_OFFSET: u32 = ${DEBUG_OFFSET}u;
const NAN_COUNT_OFFSET: u32 = ${NAN_COUNT_OFFSET}u;
const SS_PENALTY_SATURATED: u32 = ${SS_PENALTY_SATURATED}u;
const SS_ADJ_OVERFLOW: u32 = ${SS_ADJ_OVERFLOW}u;
const SS_UNCOLORED: u32 = ${SS_UNCOLORED}u;
const SS_HASH_OCCUPANCY: u32 = ${SS_HASH_OCCUPANCY}u;

struct CapacityLayout {
    csrDataOffset: u32,
    csrOffsetsOffset: u32,
    csrHeadsOffset: u32,
    adjOffset: u32,
    csBase: u32,
    sortedOffset: u32,
    colorMetaOffset: u32,
}

fn getLayout() -> CapacityLayout {
    let cap = params.capacity;
    let mc = cap * params.constraintMul;
    let hc = cap * params.hashMul;
    let cdb = HASH_BASE + hc;
    let csrOff = cdb + mc * 2u;
    let csrHeads = csrOff + cap + 1u;
    let adj = csrHeads + cap;
    let cgSize = adj - cdb + cap * ADJ_STRIDE;
    return CapacityLayout(
        cdb,
        csrOff, csrHeads, adj,
        cdb + cgSize,
        cap, cap * 2u,
    );
}

`;

const solverMathWGSL = /* wgsl */ `
${coloringConstantsWGSL}

fn quatNormalize(q: vec4f) -> vec4f {
    let len = length(q);
    if (len < 1e-12) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    return q / len;
}

fn quatIntegrate(q: vec4f, v: vec3f) -> vec4f {
    let dq = vec4f(v, 0.0);
    let prod = quatMul(dq, q);
    return quatNormalize(q + prod * 0.5);
}

fn quatInv(q: vec4f) -> vec4f {
    let ls = dot(q, q);
    return vec4f(-q.xyz, q.w) / ls;
}

fn angDispFromInitial(quat: vec4f, initialQuat: vec4f) -> vec3f {
    let dq = quatMul(quat, quatInv(initialQuat));
    return 2.0 * dq.xyz;
}

fn solve6(a: array<f32, 36>, b: array<f32, 6>) -> array<f32, 6> {
    let D0 = max(a[0], 1e-20);
    let L10 = a[6] / D0;
    let L20 = a[12] / D0;
    let L30 = a[18] / D0;
    let L40 = a[24] / D0;
    let L50 = a[30] / D0;

    let D1 = max(a[7] - L10 * L10 * D0, 1e-20);
    let L21 = (a[13] - L20 * L10 * D0) / D1;
    let L31 = (a[19] - L30 * L10 * D0) / D1;
    let L41 = (a[25] - L40 * L10 * D0) / D1;
    let L51 = (a[31] - L50 * L10 * D0) / D1;

    let D2 = max(a[14] - (L20 * L20 * D0 + L21 * L21 * D1), 1e-20);
    let L32 = (a[20] - L30 * L20 * D0 - L31 * L21 * D1) / D2;
    let L42 = (a[26] - L40 * L20 * D0 - L41 * L21 * D1) / D2;
    let L52 = (a[32] - L50 * L20 * D0 - L51 * L21 * D1) / D2;

    let D3 = max(a[21] - ((L30 * L30 * D0 + L31 * L31 * D1) + L32 * L32 * D2), 1e-20);
    let L43 = (a[27] - L40 * L30 * D0 - L41 * L31 * D1 - L42 * L32 * D2) / D3;
    let L53 = (a[33] - L50 * L30 * D0 - L51 * L31 * D1 - L52 * L32 * D2) / D3;

    let D4 = max(a[28] - (((L40 * L40 * D0 + L41 * L41 * D1) + L42 * L42 * D2) + L43 * L43 * D3), 1e-20);
    let L54 = (a[34] - L50 * L40 * D0 - L51 * L41 * D1 - L52 * L42 * D2 - L53 * L43 * D3) / D4;

    let D5 = max(a[35] - ((((L50 * L50 * D0 + L51 * L51 * D1) + L52 * L52 * D2) + L53 * L53 * D3) + L54 * L54 * D4), 1e-20);

    var y: array<f32, 6>;
    y[0] = b[0];
    y[1] = b[1] - L10 * y[0];
    y[2] = b[2] - L20 * y[0] - L21 * y[1];
    y[3] = b[3] - L30 * y[0] - L31 * y[1] - L32 * y[2];
    y[4] = b[4] - L40 * y[0] - L41 * y[1] - L42 * y[2] - L43 * y[3];
    y[5] = b[5] - L50 * y[0] - L51 * y[1] - L52 * y[2] - L53 * y[3] - L54 * y[4];

    y[0] /= D0;
    y[1] /= D1;
    y[2] /= D2;
    y[3] /= D3;
    y[4] /= D4;
    y[5] /= D5;

    var x: array<f32, 6>;
    x[5] = y[5];
    x[4] = y[4] - L54 * x[5];
    x[3] = y[3] - L43 * x[4] - L53 * x[5];
    x[2] = y[2] - L32 * x[3] - L42 * x[4] - L52 * x[5];
    x[1] = y[1] - L21 * x[2] - L31 * x[3] - L41 * x[4] - L51 * x[5];
    x[0] = y[0] - L10 * x[1] - L20 * x[2] - L30 * x[3] - L40 * x[4] - L50 * x[5];

    return x;
}

fn addJacobianToSystem(lhs: ptr<function, array<f32, 36>>, rhs: ptr<function, array<f32, 6>>, n: vec3f, rxn: vec3f, f: f32, pen: f32, G: array<f32, 6>) {
    let J0 = n.x; let J1 = n.y; let J2 = n.z;
    let J3 = rxn.x; let J4 = rxn.y; let J5 = rxn.z;

    (*rhs)[0] += J0 * f;
    (*rhs)[1] += J1 * f;
    (*rhs)[2] += J2 * f;
    (*rhs)[3] += J3 * f;
    (*rhs)[4] += J4 * f;
    (*rhs)[5] += J5 * f;

    (*lhs)[0]  += J0 * J0 * pen + G[0];
    (*lhs)[6]  += J1 * J0 * pen;
    (*lhs)[7]  += J1 * J1 * pen + G[1];
    (*lhs)[12] += J2 * J0 * pen;
    (*lhs)[13] += J2 * J1 * pen;
    (*lhs)[14] += J2 * J2 * pen + G[2];

    (*lhs)[18] += J3 * J0 * pen;
    (*lhs)[19] += J3 * J1 * pen;
    (*lhs)[20] += J3 * J2 * pen;
    (*lhs)[21] += J3 * J3 * pen + G[3];
    (*lhs)[24] += J4 * J0 * pen;
    (*lhs)[25] += J4 * J1 * pen;
    (*lhs)[26] += J4 * J2 * pen;
    (*lhs)[27] += J4 * J3 * pen;
    (*lhs)[28] += J4 * J4 * pen + G[4];
    (*lhs)[30] += J5 * J0 * pen;
    (*lhs)[31] += J5 * J1 * pen;
    (*lhs)[32] += J5 * J2 * pen;
    (*lhs)[33] += J5 * J3 * pen;
    (*lhs)[34] += J5 * J4 * pen;
    (*lhs)[35] += J5 * J5 * pen + G[5];
}

fn accumulateContact(
    lhs: ptr<function, array<f32, 36>>,
    rhs: ptr<function, array<f32, 6>>,
    jac: Jacobians,
    F: vec3f,
    pen_n: f32, pen_t1: f32, pen_t2: f32,
) {
    let K = vec3f(pen_n, pen_t1, pen_t2);
    let jLT0 = vec3f(jac.J_n.x, jac.J_t1.x, jac.J_t2.x);
    let jLT1 = vec3f(jac.J_n.y, jac.J_t1.y, jac.J_t2.y);
    let jLT2 = vec3f(jac.J_n.z, jac.J_t1.z, jac.J_t2.z);
    let jAT0 = vec3f(jac.rxn_n.x, jac.rxn_t1.x, jac.rxn_t2.x);
    let jAT1 = vec3f(jac.rxn_n.y, jac.rxn_t1.y, jac.rxn_t2.y);
    let jAT2 = vec3f(jac.rxn_n.z, jac.rxn_t1.z, jac.rxn_t2.z);
    let jLTK0 = jLT0 * K;
    let jLTK1 = jLT1 * K;
    let jLTK2 = jLT2 * K;
    let jATK0 = jAT0 * K;
    let jATK1 = jAT1 * K;
    let jATK2 = jAT2 * K;

    (*lhs)[0]  += dot(jLTK0, jLT0);
    (*lhs)[6]  += dot(jLTK1, jLT0);
    (*lhs)[7]  += dot(jLTK1, jLT1);
    (*lhs)[12] += dot(jLTK2, jLT0);
    (*lhs)[13] += dot(jLTK2, jLT1);
    (*lhs)[14] += dot(jLTK2, jLT2);

    (*lhs)[21] += dot(jATK0, jAT0);
    (*lhs)[27] += dot(jATK1, jAT0);
    (*lhs)[28] += dot(jATK1, jAT1);
    (*lhs)[33] += dot(jATK2, jAT0);
    (*lhs)[34] += dot(jATK2, jAT1);
    (*lhs)[35] += dot(jATK2, jAT2);

    (*lhs)[18] += dot(jATK0, jLT0);
    (*lhs)[19] += dot(jATK0, jLT1);
    (*lhs)[20] += dot(jATK0, jLT2);
    (*lhs)[24] += dot(jATK1, jLT0);
    (*lhs)[25] += dot(jATK1, jLT1);
    (*lhs)[26] += dot(jATK1, jLT2);
    (*lhs)[30] += dot(jATK2, jLT0);
    (*lhs)[31] += dot(jATK2, jLT1);
    (*lhs)[32] += dot(jATK2, jLT2);

    (*rhs)[0] += jLT0.x * F.x + jLT0.y * F.y + jLT0.z * F.z;
    (*rhs)[1] += jLT1.x * F.x + jLT1.y * F.y + jLT1.z * F.z;
    (*rhs)[2] += jLT2.x * F.x + jLT2.y * F.y + jLT2.z * F.z;
    (*rhs)[3] += jAT0.x * F.x + jAT0.y * F.y + jAT0.z * F.z;
    (*rhs)[4] += jAT1.x * F.x + jAT1.y * F.y + jAT1.z * F.z;
    (*rhs)[5] += jAT2.x * F.x + jAT2.y * F.y + jAT2.z * F.z;
}

fn applyBallJointDirect(
    lhs: ptr<function, array<f32, 36>>,
    rhs: ptr<function, array<f32, 6>>,
    con: GPUConstraint, idx: u32,
) {
    let bA = bodies[con.bodyA];
    let bB = bodies[u32(con.bodyB)];

    let rAw = quatRotate(bA.quat, con.rA);
    let rBw = quatRotate(bB.quat, con.rB);

    var C = (bA.pos + rAw) - (bB.pos + rBw);
    if (con.stiffness >= 1e30) {
        C -= vec3f(con.C_init_n, con.C_init_t1, con.C_init_t2) * params.alpha;
    }

    let K = vec3f(con.penalty_n, con.penalty_t1, con.penalty_t2);
    let F = K * C + vec3f(con.lambda_n, con.lambda_t1, con.lambda_t2);

    let isA = idx == con.bodyA;
    let s = select(-1.0, 1.0, isA);
    let rWorld = quatRotate(bodies[idx].quat, select(con.rB, con.rA, isA));
    let angArm = select(rWorld, -rWorld, isA);

    (*lhs)[0]  += K.x;
    (*lhs)[7]  += K.y;
    (*lhs)[14] += K.z;

    let rx = angArm.x; let ry = angArm.y; let rz = angArm.z;

    (*lhs)[21] += K.y * rz * rz + K.z * ry * ry;
    (*lhs)[27] += -K.z * rx * ry;
    (*lhs)[28] += K.x * rz * rz + K.z * rx * rx;
    (*lhs)[33] += -K.y * rx * rz;
    (*lhs)[34] += -K.x * ry * rz;
    (*lhs)[35] += K.x * ry * ry + K.y * rx * rx;

    (*lhs)[18] += 0.0;            (*lhs)[19] += rz * K.y * s;   (*lhs)[20] += -ry * K.z * s;
    (*lhs)[24] += -rz * K.x * s;  (*lhs)[25] += 0.0;            (*lhs)[26] += rx * K.z * s;
    (*lhs)[30] += ry * K.x * s;   (*lhs)[31] += -rx * K.y * s;  (*lhs)[32] += 0.0;

    let geoArm = select(-rWorld, rWorld, isA);
    let Hc0 = vec3f(-(F.y * geoArm.y + F.z * geoArm.z), F.x * geoArm.y, F.x * geoArm.z);
    let Hc1 = vec3f(F.y * geoArm.x, -(F.x * geoArm.x + F.z * geoArm.z), F.y * geoArm.z);
    let Hc2 = vec3f(F.z * geoArm.x, F.z * geoArm.y, -(F.x * geoArm.x + F.y * geoArm.y));
    (*lhs)[21] += length(Hc0);
    (*lhs)[28] += length(Hc1);
    (*lhs)[35] += length(Hc2);

    (*rhs)[0] += s * F.x;
    (*rhs)[1] += s * F.y;
    (*rhs)[2] += s * F.z;
    let angF = -cross(angArm, F);
    (*rhs)[3] += angF.x;
    (*rhs)[4] += angF.y;
    (*rhs)[5] += angF.z;
}


`;

const bodyColsHelpersWGSL = /* wgsl */ `
const COL_POS: u32 = ${BODY_COL_POS}u;
const COL_INITIAL: u32 = ${BODY_COL_INITIAL}u;
const COL_CUMANG: u32 = ${BODY_COL_CUMANG}u;
const COL_QUAT: u32 = ${BODY_COL_QUAT}u;
const COL_INITIAL_QUAT: u32 = ${BODY_COL_INITIAL_QUAT}u;

fn colIdx(col: u32, i: u32) -> u32 {
    return col * params.capacity + i;
}

fn contactArmFields(rA: vec3f, rB: vec3f, normal: vec3f, isA: bool, quat: vec4f, initialQuat: vec4f, radius: f32, collType: f32) -> vec3f {
    let s = select(1.0, -1.0, isA);
    let radial = s * normal * radius;
    let localArm = select(rB, rA, isA);
    let isSphere = collType == SOLVER_SHAPE_SPHERE;
    let isCapsule = collType == SOLVER_SHAPE_CAPSULE;
    let iqc = vec4f(-initialQuat.xyz, initialQuat.w);
    let radialLocal = quatRotate(iqc, radial);
    let capsuleArm = quatRotate(quat, localArm - radialLocal) + radial;
    let boxArm = quatRotate(quat, localArm);
    let arm = select(boxArm, capsuleArm, isCapsule);
    return select(arm, radial, isSphere);
}
`;

export const solverWGSL = /* wgsl */ `
${sharedSolverWGSL}
${solverMathWGSL}
${bodyColsHelpersWGSL}

@compute @workgroup_size(64)
fn syncBodyCols(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.bodyCount) { return; }
    let body = bodies[idx];
    bodyCols[colIdx(COL_POS, idx)]          = vec4f(body.pos, body.radius);
    bodyCols[colIdx(COL_INITIAL, idx)]      = vec4f(body.initial, body.colliderType);
    bodyCols[colIdx(COL_CUMANG, idx)]       = vec4f(body.cumAng, 0.0);
    bodyCols[colIdx(COL_QUAT, idx)]         = body.quat;
    bodyCols[colIdx(COL_INITIAL_QUAT, idx)] = body.initialQuat;
}

fn detectBallJoint(ji: u32) {
    let joint = joints[ji];
    let bodyA = bodies[joint.bodyA];
    let bodyB = bodies[joint.bodyB];

    let rAw = quatRotate(bodyA.quat, joint.localAnchorA);
    let rBw = quatRotate(bodyB.quat, joint.localAnchorB);

    let worldA = bodyA.pos + rAw;
    let worldB = bodyB.pos + rBw;
    let diff_init = worldA - worldB;

    let featureKey = 0x80000000u + ji;
    let wsKey = 0x80000000u | ji;

    let stiffness = select(1e30, joint.stiffness, joint.stiffness > 0.0);

    pushConstraint(
        joint.bodyA, i32(joint.bodyB), featureKey,
        vec3f(1.0, 0.0, 0.0), diff_init.x,
        vec3f(0.0, 1.0, 0.0), diff_init.y,
        vec3f(0.0, 0.0, 1.0), diff_init.z,
        joint.localAnchorA, joint.localAnchorB,
        0.0,
        wsKey, 1u,
        -1e30, 1e30, stiffness,
    );
}

fn detectSpring(ji: u32) {
    let joint = joints[ji];
    let bodyA = bodies[joint.bodyA];
    let bodyB = bodies[joint.bodyB];

    let rA = quatRotate(bodyA.quat, joint.localAnchorA);
    let rB = quatRotate(bodyB.quat, joint.localAnchorB);

    let worldA = bodyA.pos + rA;
    let worldB = bodyB.pos + rB;
    let diff = worldB - worldA;
    let dist = length(diff);
    let restLength = joint.restLength;
    let stiffness = joint.stiffness;

    var normal = vec3f(1.0, 0.0, 0.0);
    if (dist > 1e-8) {
        normal = diff / dist;
    }

    let C_n = dist - restLength;
    let featureKey = 0x90000000u + ji;
    let t = tangentBasis(normal);
    let wsKey = 0x90000000u | ji;

    pushConstraint(
        joint.bodyA, i32(joint.bodyB), featureKey,
        normal, C_n,
        t[0], 0.0, t[1], 0.0,
        rA, rB,
        0.0,
        wsKey, 2u,
        -1e30, 1e30, stiffness,
    );
}

fn computeBallJointCFn(c: GPUConstraint, ji: u32, currentAlpha: f32) -> vec3f {
    let bodyA = bodies[c.bodyA];
    let bodyB = bodies[u32(c.bodyB)];

    let rA = quatRotate(bodyA.quat, c.rA);
    let rB = quatRotate(bodyB.quat, c.rB);

    let worldA = bodyA.pos + rA;
    let worldB = bodyB.pos + rB;
    let diff = worldA - worldB;

    let Cn_n = dot(c.normal, diff);
    let Cn_t1 = dot(c.tangent1, diff);
    let Cn_t2 = dot(c.tangent2, diff);

    return vec3f(
        Cn_n - currentAlpha * c.C_init_n,
        Cn_t1 - currentAlpha * c.C_init_t1,
        Cn_t2 - currentAlpha * c.C_init_t2,
    );
}

fn computeSpringCFn(c: GPUConstraint, ji: u32) -> vec3f {
    let joint = joints[ji];
    let bodyA = bodies[c.bodyA];
    let bodyB = bodies[u32(c.bodyB)];

    let rA = quatRotate(bodyA.quat, joint.localAnchorA);
    let rB = quatRotate(bodyB.quat, joint.localAnchorB);

    let worldA = bodyA.pos + rA;
    let worldB = bodyB.pos + rB;
    let diff = worldB - worldA;
    let dist = length(diff);

    return vec3f(dist - joint.restLength, 0.0, 0.0);
}

fn contactArm(c: GPUConstraint, bodyIdx: u32) -> vec3f {
    let body = bodies[bodyIdx];
    let isA = bodyIdx == c.bodyA;
    let s = select(1.0, -1.0, isA);
    let radial = s * c.normal * body.radius;
    let localArm = select(c.rB, c.rA, isA);
    let isSphere = body.colliderType == SOLVER_SHAPE_SPHERE;
    let isCapsule = body.colliderType == SOLVER_SHAPE_CAPSULE;
    let iqc = vec4f(-body.initialQuat.xyz, body.initialQuat.w);
    let radialLocal = quatRotate(iqc, radial);
    let capsuleArm = quatRotate(body.quat, localArm - radialLocal) + radial;
    let boxArm = quatRotate(body.quat, localArm);
    let arm = select(boxArm, capsuleArm, isCapsule);
    return select(arm, radial, isSphere);
}

fn computeConstraintC(c: GPUConstraint, currentAlpha: f32) -> vec3f {
    let bA = bodies[c.bodyA];
    let bB = bodies[u32(c.bodyB)];
    let dqALin = bA.pos - bA.initial;
    let dqAAng = bA.cumAng;
    let dqBLin = bB.pos - bB.initial;
    let dqBAng = bB.cumAng;
    let rAW = contactArm(c, c.bodyA);
    let rBW = contactArm(c, u32(c.bodyB));
    let oneMinusAlpha = 1.0 - currentAlpha;

    let jALin_n = c.normal;
    let jBLin_n = -c.normal;
    let jAAng_n = cross(rAW, jALin_n);
    let jBAng_n = cross(rBW, jBLin_n);
    let C_n = oneMinusAlpha * c.C_init_n + dot(jALin_n, dqALin) + dot(jBLin_n, dqBLin) + dot(jAAng_n, dqAAng) + dot(jBAng_n, dqBAng);

    let jALin_t1 = c.tangent1;
    let jBLin_t1 = -c.tangent1;
    let jAAng_t1 = cross(rAW, jALin_t1);
    let jBAng_t1 = cross(rBW, jBLin_t1);
    let C_t1 = oneMinusAlpha * c.C_init_t1 + dot(jALin_t1, dqALin) + dot(jBLin_t1, dqBLin) + dot(jAAng_t1, dqAAng) + dot(jBAng_t1, dqBAng);

    let jALin_t2 = c.tangent2;
    let jBLin_t2 = -c.tangent2;
    let jAAng_t2 = cross(rAW, jALin_t2);
    let jBAng_t2 = cross(rBW, jBLin_t2);
    let C_t2 = oneMinusAlpha * c.C_init_t2 + dot(jALin_t2, dqALin) + dot(jBLin_t2, dqBLin) + dot(jAAng_t2, dqAAng) + dot(jBAng_t2, dqBAng);

    return vec3f(C_n, C_t1, C_t2);
}

struct Jacobians {
    J_n: vec3f,
    rxn_n: vec3f,
    J_t1: vec3f,
    rxn_t1: vec3f,
    J_t2: vec3f,
    rxn_t2: vec3f,
}

fn applySpringDirect(lhs: ptr<function, array<f32, 36>>, rhs: ptr<function, array<f32, 6>>, c: GPUConstraint, idx: u32) {
    let ji = c.featureKey - 0x90000000u;
    let joint = joints[ji];
    let sBodyA = bodies[c.bodyA];
    let sBodyB = bodies[u32(c.bodyB)];
    let srA = quatRotate(sBodyA.quat, joint.localAnchorA);
    let srB = quatRotate(sBodyB.quat, joint.localAnchorB);
    let sDiff = (sBodyA.pos + srA) - (sBodyB.pos + srB);
    let sDist = length(sDiff);
    if (sDist <= 1e-6) { return; }
    let sNormal = sDiff / sDist;
    let springC = sDist - joint.restLength;
    let springF = joint.stiffness * springC;
    let isA = idx == c.bodyA;
    let sSign = select(-1.0, 1.0, isA);
    let sArm = select(srB, srA, isA);
    let sJ_n = sNormal * sSign;
    let sRxn_n = cross(sArm, sJ_n);
    addJacobianToSystem(lhs, rhs, sJ_n, sRxn_n, springF, joint.stiffness, G_ZERO);
}

fn computeCForType(c: GPUConstraint, currentAlpha: f32) -> vec3f {
    if (c.bilateral == CONSTRAINT_SPRING) {
        let ji = c.featureKey - 0x90000000u;
        return computeSpringCFn(c, ji);
    }
    if (c.bilateral == CONSTRAINT_BALL) {
        let ji = c.featureKey - 0x80000000u;
        return computeBallJointCFn(c, ji, currentAlpha);
    }
    let alpha = select(currentAlpha, 0.0, c.bilateral == CONSTRAINT_KINEMATIC);
    return computeConstraintC(c, alpha);
}

@compute @workgroup_size(64)
fn warmstartBodies(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.bodyCount) { return; }

    let dt = params.dt;
    let dt2 = dt * dt;

    var body = bodies[idx];
    let g = vec3f(0.0, params.gravity * body.gravity, 0.0);

    if (body.pos.x != body.pos.x || body.pos.y != body.pos.y || body.pos.z != body.pos.z) {
        atomicAdd(&solverState[NAN_COUNT_OFFSET], 1u);
    }

    if (body.mass <= 0.0) {
        body.initial = body.pos;
        body.initialQuat = body.quat;
        bodies[idx] = body;
        return;
    }

    let fOff = idx * 8u;
    let extForce = vec3f(forces[fOff], forces[fOff + 1u], forces[fOff + 2u]);
    let extTorque = vec3f(forces[fOff + 3u], forces[fOff + 4u], forces[fOff + 5u]);
    let forceMode = forces[fOff + 6u];

    if (forceMode > 0.5) {
        body.vel = extForce;
        body.angVel = extTorque;
    } else {
        body.vel += (extForce / body.mass) * dt;

        let invI = vec3f(1.0 / body.momentX, 1.0 / body.momentY, 1.0 / body.momentZ);
        let q = body.quat;
        let tLocal = quatRotate(quatInv(q), extTorque);
        body.angVel += invI * tLocal * dt;
    }

    let angSpeedPre = length(body.angVel);
    if (angSpeedPre > MAX_ANGVEL) {
        body.angVel = body.angVel * (MAX_ANGVEL / angSpeedPre);
    }

    body.inertial = body.pos + body.vel * dt + g * dt2;
    body.inertialQuat = quatIntegrate(body.quat, body.angVel * dt);

    let accel = (body.vel - body.prevVel) / dt;
    let accelExt = accel.y * sign(params.gravity);
    var accelWeight = clamp(accelExt / abs(params.gravity), 0.0, 1.0);
    if (accelWeight != accelWeight) { accelWeight = 0.0; }

    body.initial = body.pos;
    body.initialQuat = body.quat;

    body.pos = body.pos + body.vel * dt + g * (accelWeight * dt * dt);
    body.quat = body.inertialQuat;

    bodies[idx] = body;
}

@compute @workgroup_size(64)
fn detectJoints(@builtin(global_invocation_id) gid: vec3u) {
    let ji = gid.x;
    if (ji >= params.jointCount) { return; }
    if (joints[ji].broken != 0u) { return; }
    if (joints[ji].jointType == 1u) {
        detectSpring(ji);
    } else {
        detectBallJoint(ji);
    }
}


@compute @workgroup_size(64)
fn initBodyCache(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.bodyCount) { return; }
    let body = bodies[idx];
    if (body.mass <= 0.0) {
        bodies[idx].cumAng = vec3f(0.0);
    } else {
        bodies[idx].cumAng = angDispFromInitial(body.quat, body.initialQuat);
    }
}

@compute @workgroup_size(64)
fn cacheContactC(@builtin(global_invocation_id) gid: vec3u) {
    let ci = gid.x;
    let numConstraints = atomicLoad(&solverState[SS_CONSTRAINT_COUNT]);
    if (ci >= numConstraints) { return; }
    let c = constraints[ci];
    if (c.bilateral != CONSTRAINT_CONTACT && c.bilateral != CONSTRAINT_KINEMATIC) { return; }
    let rAW = quatRotate(bodies[c.bodyA].initialQuat, c.rA);
    let rBW = quatRotate(bodies[u32(c.bodyB)].initialQuat, c.rB);
    constraints[ci].rAW = rAW;
    constraints[ci].rBW_x = rBW.x;
    constraints[ci].rBW_y = rBW.y;
    constraints[ci].rBW_z = rBW.z;
}

@compute @workgroup_size(64)
fn solveDual(@builtin(global_invocation_id) gid: vec3u) {
    let ci = gid.x;
    let numConstraints = atomicLoad(&solverState[SS_CONSTRAINT_COUNT]);
    if (ci >= numConstraints) { return; }

    var c = constraints[ci];
    if (c.bilateral == CONSTRAINT_SPRING) {
        constraints[ci] = c;
        return;
    }

    let penCap = min(PENALTY_MAX, c.stiffness);
    let beta = params.betaLin;

    if (c.bilateral == CONSTRAINT_BALL) {
        let bji = c.featureKey - 0x80000000u;
        let bA = bodies[c.bodyA];
        let bB = bodies[u32(c.bodyB)];
        let rAw = quatRotate(bA.quat, c.rA);
        let rBw = quatRotate(bB.quat, c.rB);
        var C = (bA.pos + rAw) - (bB.pos + rBw);
        if (c.stiffness >= 1e30) {
            C -= vec3f(c.C_init_n, c.C_init_t1, c.C_init_t2) * params.alpha;
            let K = vec3f(c.penalty_n, c.penalty_t1, c.penalty_t2);
            let newLambda = K * C + vec3f(c.lambda_n, c.lambda_t1, c.lambda_t2);
            c.lambda_n = newLambda.x;
            c.lambda_t1 = newLambda.y;
            c.lambda_t2 = newLambda.z;
        }
        c.penalty_n = min(c.penalty_n + beta * abs(C.x), penCap);
        c.penalty_t1 = min(c.penalty_t1 + beta * abs(C.y), penCap);
        c.penalty_t2 = min(c.penalty_t2 + beta * abs(C.z), penCap);
        c.stick = 0u;

        let frac = joints[bji].fracture;
        if (frac > 0.0) {
            let forceSq = c.lambda_n * c.lambda_n + c.lambda_t1 * c.lambda_t1 + c.lambda_t2 * c.lambda_t2;
            if (forceSq > frac * frac) {
                c.lambda_n = 0.0;
                c.lambda_t1 = 0.0;
                c.lambda_t2 = 0.0;
                c.penalty_n = 0.0;
                c.penalty_t1 = 0.0;
                c.penalty_t2 = 0.0;
                joints[bji].broken = 1u;
            }
        }
    } else {
        let Cs = computeCForType(c, params.alpha);
        let lambda_used = select(0.0, c.lambda_n, c.stiffness >= 1e30);
        c.lambda_n = clamp(c.penalty_n * Cs.x + lambda_used, c.fmin_n, c.fmax_n);
        if (c.lambda_n > c.fmin_n && c.lambda_n < c.fmax_n) {
            c.penalty_n = min(c.penalty_n + beta * abs(Cs.x), penCap);
        }
        if (c.penalty_n >= penCap) {
            atomicAdd(&solverState[SS_PENALTY_SATURATED], 1u);
        }

        if (c.friction > 0.0) {
            let dualBound = abs(c.lambda_n) * c.friction;
            let lambda_t1_used = select(0.0, c.lambda_t1, c.stiffness >= 1e30);
            let lambda_t2_used = select(0.0, c.lambda_t2, c.stiffness >= 1e30);
            var f_t1 = c.penalty_t1 * Cs.y + lambda_t1_used;
            var f_t2 = c.penalty_t2 * Cs.z + lambda_t2_used;
            let fScale = sqrt(f_t1 * f_t1 + f_t2 * f_t2);
            if (fScale > dualBound && fScale > 0.0) {
                let ratio = dualBound / fScale;
                f_t1 *= ratio;
                f_t2 *= ratio;
            }
            c.lambda_t1 = f_t1;
            c.lambda_t2 = f_t2;
            if (fScale <= dualBound) {
                c.penalty_t1 = min(c.penalty_t1 + beta * abs(Cs.y), penCap);
                c.penalty_t2 = min(c.penalty_t2 + beta * abs(Cs.z), penCap);
                c.stick = select(0u, 1u, sqrt(Cs.y * Cs.y + Cs.z * Cs.z) < STICK_THRESH);
            }
        }

    }

    constraints[ci] = c;
}

@compute @workgroup_size(1)
fn advanceIteration(@builtin(global_invocation_id) gid: vec3u) {
    atomicAdd(&solverState[SS_ITERATION], 1u);
}

@compute @workgroup_size(64)
fn computeVelocities(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.bodyCount) { return; }

    let dt = params.dt;
    var body = bodies[idx];

    if (body.mass <= 0.0) { return; }

    body.prevVel = body.vel;
    body.prevAngVel = body.angVel;
    body.vel = (body.pos - body.initial) / dt;

    let dqFinal = quatMul(body.quat, quatInv(body.initialQuat));
    body.angVel = 2.0 * dqFinal.xyz / dt;
    bodies[idx] = body;
}

@compute @workgroup_size(64)
fn writebackWarmstarts(@builtin(global_invocation_id) gid: vec3u) {
    let ci = gid.x;
    let numConstraints = atomicLoad(&solverState[SS_CONSTRAINT_COUNT]);
    if (ci >= numConstraints) { return; }

    let c = constraints[ci];
    let slot = hashInsert(c.warmstartIdx);
    if (slot >= params.capacity * params.hashMul) { atomicAdd(&solverState[SS_HASH_OVERFLOW], 1u); return; }
    atomicAdd(&solverState[SS_HASH_OCCUPANCY], 1u);
    warmstarts[slot] = WarmstartEntry(
        c.lambda_n, c.penalty_n,
        c.lambda_t1, c.penalty_t1,
        c.lambda_t2, c.penalty_t2,
        c.stick, c.featureKey,
        c.rA, 0.0,
        c.rB, 0.0,
    );
}

@compute @workgroup_size(64)
fn solvePrimal(@builtin(global_invocation_id) gid: vec3u) {
    let L = getLayout();
    let currentColor = atomicLoad(&solverState[SS_CURRENT_COLOR]);
    let colorOffset = atomicLoad(&solverState[L.csBase + L.colorMetaOffset + MAX_COLORS + currentColor]);
    let nextOffset = atomicLoad(&solverState[L.csBase + L.colorMetaOffset + MAX_COLORS + currentColor + 1u]);
    let colorCount = nextOffset - colorOffset;
    if (gid.x >= colorCount) { return; }

    let idx = atomicLoad(&solverState[L.csBase + L.sortedOffset + colorOffset + gid.x]);
    var body = bodies[idx];
    if (body.mass <= 0.0) { return; }

    let dt = params.dt;
    let dt2 = dt * dt;
    let clStart = atomicLoad(&solverState[L.csrOffsetsOffset + idx]);
    let clEnd = atomicLoad(&solverState[L.csrOffsetsOffset + idx + 1u]);

    var lhs: array<f32, 36>;
    let mdt2 = body.mass / dt2;
    lhs[0] = mdt2;
    lhs[7] = mdt2;
    lhs[14] = mdt2;
    lhs[21] = body.momentX / dt2;
    lhs[28] = body.momentY / dt2;
    lhs[35] = body.momentZ / dt2;

    let dp = body.pos - body.inertial;
    var rhs: array<f32, 6>;
    rhs[0] = mdt2 * dp.x;
    rhs[1] = mdt2 * dp.y;
    rhs[2] = mdt2 * dp.z;

    let dq = quatMul(body.quat, quatInv(body.inertialQuat));
    let angDisp2 = 2.0 * dq.xyz;
    rhs[3] = body.momentX / dt2 * angDisp2.x;
    rhs[4] = body.momentY / dt2 * angDisp2.y;
    rhs[5] = body.momentZ / dt2 * angDisp2.z;

    for (var k = clStart; k < clEnd; k++) {
        let ci = atomicLoad(&solverState[L.csrDataOffset + k]);
        var c = constraints[ci];
        if (c.bodyA != idx && c.bodyB != i32(idx)) { continue; }

        if (c.bilateral == CONSTRAINT_SPRING) {
            applySpringDirect(&lhs, &rhs, c, idx);
            continue;
        }

        if (c.bilateral == CONSTRAINT_BALL) {
            applyBallJointDirect(&lhs, &rhs, c, idx);
            continue;
        }

        let bA_isOwn = idx == c.bodyA;
        let neighborIdx = select(c.bodyA, u32(c.bodyB), bA_isOwn);

        let nPosCol         = bodyCols[colIdx(COL_POS, neighborIdx)];
        let nInitialCol     = bodyCols[colIdx(COL_INITIAL, neighborIdx)];
        let nCumAngCol      = bodyCols[colIdx(COL_CUMANG, neighborIdx)];
        let nQuat           = bodyCols[colIdx(COL_QUAT, neighborIdx)];
        let nInitialQuat    = bodyCols[colIdx(COL_INITIAL_QUAT, neighborIdx)];

        let quatA           = select(nQuat,        body.quat,        bA_isOwn);
        let quatB           = select(body.quat,    nQuat,            bA_isOwn);
        let initialQuatA    = select(nInitialQuat, body.initialQuat, bA_isOwn);
        let initialQuatB    = select(body.initialQuat, nInitialQuat, bA_isOwn);
        let radiusA         = select(nPosCol.w,    body.radius,      bA_isOwn);
        let radiusB         = select(body.radius,  nPosCol.w,        bA_isOwn);
        let collA           = select(nInitialCol.w, body.colliderType, bA_isOwn);
        let collB           = select(body.colliderType, nInitialCol.w, bA_isOwn);

        let rAW = contactArmFields(c.rA, c.rB, c.normal, true,  quatA, initialQuatA, radiusA, collA);
        let rBW = contactArmFields(c.rA, c.rB, c.normal, false, quatB, initialQuatB, radiusB, collB);

        let posA    = select(nPosCol.xyz,     body.pos,     bA_isOwn);
        let posB    = select(body.pos,        nPosCol.xyz,  bA_isOwn);
        let initA   = select(nInitialCol.xyz, body.initial, bA_isOwn);
        let initB   = select(body.initial,    nInitialCol.xyz, bA_isOwn);
        let cumAngA = select(nCumAngCol.xyz,  body.cumAng,  bA_isOwn);
        let cumAngB = select(body.cumAng,     nCumAngCol.xyz, bA_isOwn);

        let dqALin = posA - initA;
        let dqBLin = posB - initB;
        let dqAAng = cumAngA;
        let dqBAng = cumAngB;

        let alphaUsed = select(params.alpha, 0.0, c.bilateral == CONSTRAINT_KINEMATIC);
        let oneMinusAlpha = 1.0 - alphaUsed;

        let jALin_n = c.normal;
        let jBLin_n = -c.normal;
        let jAAng_n = cross(rAW, jALin_n);
        let jBAng_n = cross(rBW, jBLin_n);
        let C_n = oneMinusAlpha * c.C_init_n + dot(jALin_n, dqALin) + dot(jBLin_n, dqBLin) + dot(jAAng_n, dqAAng) + dot(jBAng_n, dqBAng);

        let jALin_t1 = c.tangent1;
        let jBLin_t1 = -c.tangent1;
        let jAAng_t1 = cross(rAW, jALin_t1);
        let jBAng_t1 = cross(rBW, jBLin_t1);
        let C_t1 = oneMinusAlpha * c.C_init_t1 + dot(jALin_t1, dqALin) + dot(jBLin_t1, dqBLin) + dot(jAAng_t1, dqAAng) + dot(jBAng_t1, dqBAng);

        let jALin_t2 = c.tangent2;
        let jBLin_t2 = -c.tangent2;
        let jAAng_t2 = cross(rAW, jALin_t2);
        let jBAng_t2 = cross(rBW, jBLin_t2);
        let C_t2 = oneMinusAlpha * c.C_init_t2 + dot(jALin_t2, dqALin) + dot(jBLin_t2, dqBLin) + dot(jAAng_t2, dqAAng) + dot(jBAng_t2, dqBAng);

        let Cs = vec3f(C_n, C_t1, C_t2);

        let lambda_used = select(0.0, c.lambda_n, c.stiffness >= 1e30);
        let f_n = clamp(c.penalty_n * Cs.x + lambda_used, c.fmin_n, c.fmax_n);

        let armOwn = select(rBW, rAW, bA_isOwn);
        let sJ = select(-1.0, 1.0, bA_isOwn);
        let J_n  = c.normal   * sJ;
        let J_t1 = c.tangent1 * sJ;
        let J_t2 = c.tangent2 * sJ;
        let jac = Jacobians(J_n, cross(armOwn, J_n), J_t1, cross(armOwn, J_t1), J_t2, cross(armOwn, J_t2));

        let lambda_t1_used = select(0.0, c.lambda_t1, c.stiffness >= 1e30);
        let lambda_t2_used = select(0.0, c.lambda_t2, c.stiffness >= 1e30);

        var f_t1 = c.penalty_t1 * Cs.y + lambda_t1_used;
        var f_t2 = c.penalty_t2 * Cs.z + lambda_t2_used;
        if (c.friction > 0.0) {
            let bound = abs(f_n) * c.friction;
            let fScale = sqrt(f_t1 * f_t1 + f_t2 * f_t2);
            if (fScale > bound && fScale > 0.0) {
                let ratio = bound / fScale;
                f_t1 *= ratio;
                f_t2 *= ratio;
            }
        }
        let F = vec3f(f_n, f_t1, f_t2);

        accumulateContact(&lhs, &rhs, jac, F, c.penalty_n, c.penalty_t1, c.penalty_t2);
    }

    for (var nr = 0u; nr < 6u; nr++) { rhs[nr] = -rhs[nr]; }
    let delta = solve6(lhs, rhs);
    let newPos = body.pos + vec3f(delta[0], delta[1], delta[2]);
    let newQuat = quatIntegrate(body.quat, vec3f(delta[3], delta[4], delta[5]));
    let newCumAng = body.cumAng + vec3f(delta[3], delta[4], delta[5]);
    bodies[idx].pos    = newPos;
    bodies[idx].quat   = newQuat;
    bodies[idx].cumAng = newCumAng;
    bodyCols[colIdx(COL_POS, idx)]    = vec4f(newPos, body.radius);
    bodyCols[colIdx(COL_QUAT, idx)]   = newQuat;
    bodyCols[colIdx(COL_CUMANG, idx)] = vec4f(newCumAng, 0.0);
}

@compute @workgroup_size(1)
fn advanceColor(@builtin(global_invocation_id) gid: vec3u) {
    atomicAdd(&solverState[SS_CURRENT_COLOR], 1u);
}

@compute @workgroup_size(1)
fn resetColor(@builtin(global_invocation_id) gid: vec3u) {
    atomicStore(&solverState[SS_CURRENT_COLOR], 0u);
}

`;

export const coloringWGSL = /* wgsl */ `
${solverBaseWGSL}
${coloringConstantsWGSL}

fn colorPriority(id: u32) -> u32 {
    return id * 2654435761u;
}

@compute @workgroup_size(64)
fn clearColorBuffers(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let L = getLayout();
    if (idx < MAX_COLORS) {
        atomicStore(&solverState[L.csBase + L.colorMetaOffset + idx], 0u);
        atomicStore(&solverState[L.csBase + L.colorMetaOffset + MAX_COLORS + idx], 0u);
    }
    if (idx == 0u) {
        atomicStore(&solverState[L.csBase + L.colorMetaOffset + MAX_COLORS * 2u], 0u);
        atomicStore(&solverState[SS_CURRENT_COLOR], 0u);
    }
    if (idx >= params.bodyCount) { return; }
    atomicStore(&solverState[L.adjOffset + idx * ADJ_STRIDE], 0u);
    atomicStore(&solverState[L.csrHeadsOffset + idx], 0u);
    atomicStore(&solverState[L.csBase + idx], UNCOLORED);
}

@compute @workgroup_size(64)
fn countBodyConstraints(@builtin(global_invocation_id) gid: vec3u) {
    let ci = gid.x;
    if (ci >= atomicLoad(&solverState[SS_CONSTRAINT_COUNT])) { return; }
    let L = getLayout();
    let c = constraints[ci];
    atomicAdd(&solverState[L.csrHeadsOffset + c.bodyA], 1u);
    if (c.bodyB >= 0) {
        let b = u32(c.bodyB);
        if (bodies[b].mass > 0.0) {
            atomicAdd(&solverState[L.csrHeadsOffset + b], 1u);
        }
    }
}

@compute @workgroup_size(64)
fn scatterBodyConstraints(@builtin(global_invocation_id) gid: vec3u) {
    let ci = gid.x;
    if (ci >= atomicLoad(&solverState[SS_CONSTRAINT_COUNT])) { return; }
    let L = getLayout();
    let c = constraints[ci];
    let slotA = atomicAdd(&solverState[L.csrHeadsOffset + c.bodyA], 1u);
    atomicStore(&solverState[L.csrDataOffset + slotA], ci);
    if (c.bodyB >= 0) {
        let b = u32(c.bodyB);
        if (bodies[b].mass > 0.0) {
            let slotB = atomicAdd(&solverState[L.csrHeadsOffset + b], 1u);
            atomicStore(&solverState[L.csrDataOffset + slotB], ci);
        }
    }
}

@compute @workgroup_size(64)
fn buildAdjacencyList(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.bodyCount) { return; }
    if (bodies[idx].mass <= 0.0) { return; }
    let L = getLayout();
    let clStart = atomicLoad(&solverState[L.csrOffsetsOffset + idx]);
    let clEnd = atomicLoad(&solverState[L.csrOffsetsOffset + idx + 1u]);
    var degree = 0u;
    for (var k = clStart; k < clEnd; k++) {
        let ci = atomicLoad(&solverState[L.csrDataOffset + k]);
        let c = constraints[ci];
        var neighbor = c.bodyA;
        if (neighbor == idx) {
            if (c.bodyB < 0) { continue; }
            neighbor = u32(c.bodyB);
        }
        if (neighbor == idx) { continue; }
        if (bodies[neighbor].mass <= 0.0) { continue; }
        var dup = false;
        for (var d = 0u; d < degree; d++) {
            if (atomicLoad(&solverState[L.adjOffset + idx * ADJ_STRIDE + 1u + d]) == neighbor) {
                dup = true;
                break;
            }
        }
        if (dup) { continue; }
        if (degree < MAX_DEGREE) {
            atomicStore(&solverState[L.adjOffset + idx * ADJ_STRIDE + 1u + degree], neighbor);
            degree++;
        } else {
            atomicAdd(&solverState[SS_ADJ_OVERFLOW], 1u);
        }
    }

    atomicStore(&solverState[L.adjOffset + idx * ADJ_STRIDE], degree);
}

@compute @workgroup_size(64)
fn graphColor(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.bodyCount) { return; }
    let L = getLayout();
    if (bodies[idx].mass <= 0.0) {
        atomicStore(&solverState[L.csBase + idx], KINEMATIC_COLOR);
        return;
    }
    if (atomicLoad(&solverState[L.csBase + idx]) != UNCOLORED) { return; }

    let myPriority = colorPriority(idx);
    let adjCount = min(atomicLoad(&solverState[L.adjOffset + idx * ADJ_STRIDE]), MAX_DEGREE);
    for (var i = 0u; i < adjCount; i++) {
        let neighbor = atomicLoad(&solverState[L.adjOffset + idx * ADJ_STRIDE + 1u + i]);
        if (colorPriority(neighbor) < myPriority && atomicLoad(&solverState[L.csBase + neighbor]) == UNCOLORED) {
            return;
        }
    }

    var usedColors = 0u;
    for (var i = 0u; i < adjCount; i++) {
        let neighbor = atomicLoad(&solverState[L.adjOffset + idx * ADJ_STRIDE + 1u + i]);
        let nc = atomicLoad(&solverState[L.csBase + neighbor]);
        if (nc < MAX_COLORS) {
            usedColors |= (1u << nc);
        }
    }
    atomicStore(&solverState[L.csBase + idx], countTrailingZeros(~usedColors));
}

@compute @workgroup_size(64)
fn countColors(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.bodyCount) { return; }
    let L = getLayout();
    let c = atomicLoad(&solverState[L.csBase + idx]);
    if (c < MAX_COLORS) {
        atomicAdd(&solverState[L.csBase + L.colorMetaOffset + c], 1u);
    } else if (bodies[idx].mass > 0.0) {
        atomicAdd(&solverState[SS_UNCOLORED], 1u);
        atomicAdd(&solverState[L.csBase + L.colorMetaOffset + 0u], 1u);
    }
}

@compute @workgroup_size(1)
fn prefixSumColors(@builtin(global_invocation_id) gid: vec3u) {
    let L = getLayout();
    var runningOffset = 0u;
    var usedColors = 0u;
    for (var c = 0u; c < MAX_COLORS; c++) {
        let count = atomicLoad(&solverState[L.csBase + L.colorMetaOffset + c]);
        atomicStore(&solverState[DEBUG_OFFSET + c], count);
        if (count > 0u) { usedColors++; }
        atomicStore(&solverState[L.csBase + L.colorMetaOffset + MAX_COLORS + c], runningOffset);
        atomicStore(&solverState[L.csBase + L.colorMetaOffset + c], runningOffset);
        runningOffset += count;
    }
    atomicStore(&solverState[L.csBase + L.colorMetaOffset + MAX_COLORS * 2u], runningOffset);
    atomicStore(&solverState[SS_USED_COLORS], usedColors);
}

@compute @workgroup_size(64)
fn sortBodiesByColor(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.bodyCount) { return; }
    let L = getLayout();
    var c = atomicLoad(&solverState[L.csBase + idx]);
    if (c >= MAX_COLORS && bodies[idx].mass > 0.0) {
        c = 0u;
    }
    if (c < MAX_COLORS) {
        let slot = atomicAdd(&solverState[L.csBase + L.colorMetaOffset + c], 1u);
        atomicStore(&solverState[L.csBase + L.sortedOffset + slot], idx);
    }
}
`;
