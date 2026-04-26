import {
    BVH_NODE_STRUCT_WGSL,
    BLAS_NODE_STRUCT_WGSL,
    LEAF_FLAG_WGSL,
    BLAS_TRIANGLE_STRUCT_WGSL,
    OCT_DECODE_WGSL,
    AABB_SENTINEL_WGSL,
    SAFE_INVERSE_EPSILON,
} from "./structs";

export const BVH_STRUCTS = /* wgsl */ `
${BVH_NODE_STRUCT_WGSL}

${LEAF_FLAG_WGSL}
${AABB_SENTINEL_WGSL}
const INVALID_NODE: u32 = 0xFFFFFFFFu;
const MAX_STACK_DEPTH: u32 = 24u;

${OCT_DECODE_WGSL}
`;

export const TLAS_BLAS_STRUCTS = /* wgsl */ `
${BLAS_NODE_STRUCT_WGSL}
${BLAS_TRIANGLE_STRUCT_WGSL}
`;

export const TLAS_BLAS_BINDINGS = /* wgsl */ `
@group(1) @binding(0) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(1) var<storage, read> blasNodes: array<BLASNode>;
@group(1) @binding(2) var<storage, read> blasTriIds: array<u32>;
@group(1) @binding(3) var<storage, read> blasTriangles: array<BLASTriangle>;
@group(1) @binding(4) var<storage, read> entityBlasMeta: array<u32>;
@group(1) @binding(5) var<storage, read> instanceInverses: array<mat4x4<f32>>;
`;

export const BVH_UTILS_WGSL = /* wgsl */ `
fn isLeaf(child: u32) -> bool {
    return (child & LEAF_FLAG) != 0u;
}

fn leafIndex(child: u32) -> u32 {
    return child & ~LEAF_FLAG;
}

fn safeInverse(d: f32) -> f32 {
    return select(1.0 / d, AABB_SENTINEL, abs(d) < ${SAFE_INVERSE_EPSILON});
}

fn computeInvDir(dir: vec3<f32>) -> vec3<f32> {
    return vec3(
        safeInverse(dir.x),
        safeInverse(dir.y),
        safeInverse(dir.z)
    );
}

fn intersectAABBDist(origin: vec3<f32>, invDir: vec3<f32>, nodeMin: vec3<f32>, nodeMax: vec3<f32>) -> f32 {
    let t1 = (nodeMin - origin) * invDir;
    let t2 = (nodeMax - origin) * invDir;

    let tNear = min(t1, t2);
    let tFar = max(t1, t2);

    let tEnter = max(max(tNear.x, tNear.y), tNear.z);
    let tExit = min(min(tFar.x, tFar.y), tFar.z);

    if (tEnter <= tExit && tExit >= 0.0) {
        return max(tEnter, 0.0);
    }
    return AABB_SENTINEL;
}
`;

export const ANALYTIC_SHADOW_WGSL = /* wgsl */ `
fn intersectUnitBoxShadow(ray: Ray, maxT: f32) -> bool {
    let invDir = vec3(
        select(1.0 / ray.direction.x, 1e30, abs(ray.direction.x) < 1e-8),
        select(1.0 / ray.direction.y, 1e30, abs(ray.direction.y) < 1e-8),
        select(1.0 / ray.direction.z, 1e30, abs(ray.direction.z) < 1e-8)
    );

    let t1 = (vec3(-0.5) - ray.origin) * invDir;
    let t2 = (vec3(0.5) - ray.origin) * invDir;

    let tNear = min(t1, t2);
    let tFar = max(t1, t2);

    let tEnter = max(max(tNear.x, tNear.y), tNear.z);
    let tExit = min(min(tFar.x, tFar.y), tFar.z);

    if (tEnter > tExit || tExit < 0.0) {
        return false;
    }

    let t = select(tEnter, tExit, tEnter < 0.0);
    return t > 0.0 && t < maxT;
}

fn intersectUnitSphereShadow(ray: Ray, maxT: f32) -> bool {
    let oc = ray.origin;
    let a = dot(ray.direction, ray.direction);
    let b = dot(oc, ray.direction);
    let c = dot(oc, oc) - 0.25;
    let discriminant = b * b - a * c;

    if (discriminant < 0.0) {
        return false;
    }

    let sqrtD = sqrt(discriminant);
    var t = (-b - sqrtD) / a;
    if (t <= 0.0) {
        t = (-b + sqrtD) / a;
    }
    return t > 0.0 && t < maxT;
}

fn intersectUnitPlaneShadow(ray: Ray, maxT: f32) -> bool {
    if (abs(ray.direction.y) < 1e-8) { return false; }
    let t = -ray.origin.y / ray.direction.y;
    if (t <= 0.0 || t >= maxT) { return false; }
    let hx = ray.origin.x + t * ray.direction.x;
    let hz = ray.origin.z + t * ray.direction.z;
    return hx >= -0.5 && hx <= 0.5 && hz >= -0.5 && hz <= 0.5;
}

fn intersectUnitCapsuleShadow(ray: Ray, maxT: f32) -> bool {
    let r = 0.5;
    let hh = 0.5;

    let dx = ray.direction.x;
    let dz = ray.direction.z;
    let ox = ray.origin.x;
    let oz = ray.origin.z;

    let a_cyl = dx * dx + dz * dz;
    let b_cyl = ox * dx + oz * dz;
    let c_cyl = ox * ox + oz * oz - r * r;

    if (a_cyl > 1e-8) {
        let disc = b_cyl * b_cyl - a_cyl * c_cyl;
        if (disc >= 0.0) {
            let sqrtD = sqrt(disc);
            let inv = 1.0 / a_cyl;
            for (var i = 0u; i < 2u; i++) {
                let t = (-b_cyl + select(sqrtD, -sqrtD, i == 0u)) * inv;
                if (t > 0.0 && t < maxT) {
                    let y = ray.origin.y + t * ray.direction.y;
                    if (y >= -hh && y <= hh) { return true; }
                }
            }
        }
    }

    let centers = array<f32, 2>(hh, -hh);
    for (var i = 0u; i < 2u; i++) {
        let cy = centers[i];
        let oc_cap = vec3(ox, ray.origin.y - cy, oz);
        let a_s = dot(ray.direction, ray.direction);
        let b_s = dot(oc_cap, ray.direction);
        let c_s = dot(oc_cap, oc_cap) - r * r;
        let disc = b_s * b_s - a_s * c_s;
        if (disc >= 0.0) {
            let sqrtD = sqrt(disc);
            let inv = 1.0 / a_s;
            for (var j = 0u; j < 2u; j++) {
                let t = (-b_s + select(sqrtD, -sqrtD, j == 0u)) * inv;
                if (t > 0.0 && t < maxT) {
                    let hy = ray.origin.y + t * ray.direction.y;
                    if ((i == 0u && hy >= hh) || (i == 1u && hy <= -hh)) { return true; }
                }
            }
        }
    }

    return false;
}
`;

export const ANALYTIC_INTERSECTION_WGSL = /* wgsl */ `
fn intersectUnitBox(ray: Ray, maxT: f32) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = maxT;
    result.entityId = 0u;
    result.u = 0.0;
    result.v = 0.0;
    result.normal = vec3(0.0, 1.0, 0.0);
    result.worldPos = vec3(0.0);

    let invDir = vec3(
        select(1.0 / ray.direction.x, 1e30, abs(ray.direction.x) < 1e-8),
        select(1.0 / ray.direction.y, 1e30, abs(ray.direction.y) < 1e-8),
        select(1.0 / ray.direction.z, 1e30, abs(ray.direction.z) < 1e-8)
    );

    let t1 = (vec3(-0.5) - ray.origin) * invDir;
    let t2 = (vec3(0.5) - ray.origin) * invDir;

    let tNear = min(t1, t2);
    let tFar = max(t1, t2);

    let tEnter = max(max(tNear.x, tNear.y), tNear.z);
    let tExit = min(min(tFar.x, tFar.y), tFar.z);

    if (tEnter > tExit || tExit < 0.0) {
        return result;
    }

    let t = select(tEnter, tExit, tEnter < 0.0);
    if (t <= 0.0 || t >= maxT) {
        return result;
    }

    result.hit = true;
    result.t = t;
    let hitPoint = ray.origin + t * ray.direction;
    result.worldPos = hitPoint;

    let absHit = abs(hitPoint);
    if (absHit.x > absHit.y && absHit.x > absHit.z) {
        result.normal = vec3(sign(hitPoint.x), 0.0, 0.0);
    } else if (absHit.y > absHit.z) {
        result.normal = vec3(0.0, sign(hitPoint.y), 0.0);
    } else {
        result.normal = vec3(0.0, 0.0, sign(hitPoint.z));
    }

    return result;
}

fn intersectUnitSphere(ray: Ray, maxT: f32) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = maxT;
    result.entityId = 0u;
    result.u = 0.0;
    result.v = 0.0;
    result.normal = vec3(0.0, 1.0, 0.0);
    result.worldPos = vec3(0.0);

    let oc = ray.origin;
    let a = dot(ray.direction, ray.direction);
    let b = dot(oc, ray.direction);
    let c = dot(oc, oc) - 0.25;
    let discriminant = b * b - a * c;

    if (discriminant < 0.0) {
        return result;
    }

    let sqrtD = sqrt(discriminant);
    var t = (-b - sqrtD) / a;
    if (t <= 0.0) {
        t = (-b + sqrtD) / a;
    }
    if (t <= 0.0 || t >= maxT) {
        return result;
    }

    result.hit = true;
    result.t = t;
    let hitPoint = ray.origin + t * ray.direction;
    result.worldPos = hitPoint;
    result.normal = normalize(hitPoint);

    return result;
}

fn intersectUnitPlane(ray: Ray, maxT: f32) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = maxT;
    result.entityId = 0u;
    result.u = 0.0;
    result.v = 0.0;
    result.normal = vec3(0.0, 1.0, 0.0);
    result.worldPos = vec3(0.0);

    if (abs(ray.direction.y) < 1e-8) { return result; }
    let t = -ray.origin.y / ray.direction.y;
    if (t <= 0.0 || t >= maxT) { return result; }
    let hitPoint = ray.origin + t * ray.direction;
    if (hitPoint.x < -0.5 || hitPoint.x > 0.5 || hitPoint.z < -0.5 || hitPoint.z > 0.5) {
        return result;
    }

    result.hit = true;
    result.t = t;
    result.worldPos = hitPoint;
    result.normal = select(vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0), ray.direction.y > 0.0);

    return result;
}

fn intersectUnitCapsule(ray: Ray, maxT: f32) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = maxT;
    result.entityId = 0u;
    result.u = 0.0;
    result.v = 0.0;
    result.normal = vec3(0.0, 1.0, 0.0);
    result.worldPos = vec3(0.0);

    let r = 0.5;
    let hh = 0.5;
    var bestT = maxT;

    let dx = ray.direction.x;
    let dy = ray.direction.y;
    let dz = ray.direction.z;
    let ox = ray.origin.x;
    let oy = ray.origin.y;
    let oz = ray.origin.z;

    let a_cyl = dx * dx + dz * dz;
    let b_cyl = ox * dx + oz * dz;
    let c_cyl = ox * ox + oz * oz - r * r;

    if (a_cyl > 1e-8) {
        let disc = b_cyl * b_cyl - a_cyl * c_cyl;
        if (disc >= 0.0) {
            let sqrtD = sqrt(disc);
            let inv = 1.0 / a_cyl;
            for (var i = 0u; i < 2u; i++) {
                let t = (-b_cyl + select(sqrtD, -sqrtD, i == 0u)) * inv;
                if (t > 0.0 && t < bestT) {
                    let y = oy + t * dy;
                    if (y >= -hh && y <= hh) {
                        bestT = t;
                        let hp = ray.origin + t * ray.direction;
                        result.hit = true;
                        result.t = t;
                        result.worldPos = hp;
                        result.normal = normalize(vec3(hp.x, 0.0, hp.z));
                    }
                }
            }
        }
    }

    let centers = array<f32, 2>(hh, -hh);
    for (var i = 0u; i < 2u; i++) {
        let cy = centers[i];
        let ocy = oy - cy;
        let oc_cap = vec3(ox, ocy, oz);
        let a_s = dot(ray.direction, ray.direction);
        let b_s = dot(oc_cap, ray.direction);
        let c_s = dot(oc_cap, oc_cap) - r * r;
        let disc = b_s * b_s - a_s * c_s;
        if (disc >= 0.0) {
            let sqrtD = sqrt(disc);
            let inv = 1.0 / a_s;
            for (var j = 0u; j < 2u; j++) {
                let t = (-b_s + select(sqrtD, -sqrtD, j == 0u)) * inv;
                if (t > 0.0 && t < bestT) {
                    let hy = oy + t * dy;
                    let valid = (i == 0u && hy >= hh) || (i == 1u && hy <= -hh);
                    if (valid) {
                        bestT = t;
                        let hp = ray.origin + t * ray.direction;
                        result.hit = true;
                        result.t = t;
                        result.worldPos = hp;
                        result.normal = normalize(hp - vec3(0.0, cy, 0.0));
                    }
                }
            }
        }
    }

    return result;
}
`;

export const CLOSEST_HIT_FNS = /* wgsl */ `
fn intersectBLASTriangle(ray: Ray, tri: BLASTriangle) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 0.0;
    result.entityId = 0u;
    result.u = 0.0;
    result.v = 0.0;
    result.normal = vec3(0.0, 1.0, 0.0);
    result.worldPos = vec3(0.0);

    let e1 = tri.e1;
    let e2 = tri.e2;

    let h = cross(ray.direction, e2);
    let a = dot(e1, h);

    if (a > -EPSILON && a < EPSILON) {
        return result;
    }

    let f = 1.0 / a;
    let s = ray.origin - tri.v0;
    let u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return result;
    }

    let q = cross(s, e1);
    let v = f * dot(ray.direction, q);

    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    let t = f * dot(e2, q);

    if (t > EPSILON) {
        result.hit = true;
        result.t = t;
        result.u = u;
        result.v = v;
        let w = 1.0 - u - v;
        let n0 = octDecode(tri.n0_enc);
        let n1 = octDecode(tri.n1_enc);
        let n2 = octDecode(tri.n2_enc);
        result.normal = normalize(w * n0 + u * n1 + v * n2);
        result.worldPos = ray.origin + t * ray.direction;
    }

    return result;
}

fn traceBLAS(
    ray: Ray,
    nodeOffset: u32,
    triIdOffset: u32,
    triOffset: u32,
    totalTriCount: u32,
    maxT: f32
) -> HitResult {
    var closest: HitResult;
    closest.hit = false;
    closest.t = maxT;
    closest.entityId = 0u;
    closest.u = 0.0;
    closest.v = 0.0;
    closest.normal = vec3(0.0, 1.0, 0.0);
    closest.worldPos = vec3(0.0);

    if (totalTriCount == 0u) {
        return closest;
    }

    if (totalTriCount == 1u) {
        let triIdx = blasTriIds[triIdOffset];
        let tri = blasTriangles[triOffset + triIdx];
        let hit = intersectBLASTriangle(ray, tri);
        if (hit.hit && hit.t < closest.t) {
            closest = hit;
        }
        return closest;
    }

    let invDir = computeInvDir(ray.direction);

    var stack: array<u32, MAX_STACK_DEPTH>;
    var stackPtr = 0u;

    stack[stackPtr] = 0u;
    stackPtr++;

    var iterations = 0u;
    let maxIterations = min(totalTriCount * 3u, 10000u);

    while (stackPtr > 0u && iterations < maxIterations) {
        iterations++;
        stackPtr--;
        let localIdx = stack[stackPtr];
        let node = blasNodes[nodeOffset + localIdx];

        let left = node.leftChild;
        let right = node.rightChild;

        if (isLeaf(left)) {
            let leafIdx = leafIndex(left);
            let triIdx = blasTriIds[triIdOffset + leafIdx];
            let tri = blasTriangles[triOffset + triIdx];
            let hit = intersectBLASTriangle(ray, tri);
            if (hit.hit && hit.t < closest.t) {
                closest = hit;
            }
        } else {
            let leftNode = blasNodes[nodeOffset + left];
            let dist = intersectAABBDist(ray.origin, invDir,
                vec3(leftNode.minX, leftNode.minY, leftNode.minZ),
                vec3(leftNode.maxX, leftNode.maxY, leftNode.maxZ));
            if (dist < closest.t && stackPtr < MAX_STACK_DEPTH) {
                stack[stackPtr] = left;
                stackPtr++;
            }
        }

        if (isLeaf(right)) {
            let leafIdx = leafIndex(right);
            let triIdx = blasTriIds[triIdOffset + leafIdx];
            let tri = blasTriangles[triOffset + triIdx];
            let hit = intersectBLASTriangle(ray, tri);
            if (hit.hit && hit.t < closest.t) {
                closest = hit;
            }
        } else {
            let rightNode = blasNodes[nodeOffset + right];
            let dist = intersectAABBDist(ray.origin, invDir,
                vec3(rightNode.minX, rightNode.minY, rightNode.minZ),
                vec3(rightNode.maxX, rightNode.maxY, rightNode.maxZ));
            if (dist < closest.t && stackPtr < MAX_STACK_DEPTH) {
                stack[stackPtr] = right;
                stackPtr++;
            }
        }
    }

    return closest;
}

fn trace(ray: Ray) -> HitResult {
    var closest: HitResult;
    closest.hit = false;
    closest.t = AABB_SENTINEL;
    closest.entityId = 0u;
    closest.u = 0.0;
    closest.v = 0.0;
    closest.normal = vec3(0.0, 1.0, 0.0);
    closest.worldPos = vec3(0.0);

    let count = getInstanceCount();
    if (count == 0u) {
        return closest;
    }

    let invDir = computeInvDir(ray.direction);

    var stack: array<u32, MAX_STACK_DEPTH>;
    var stackPtr = 0u;
    stack[stackPtr] = 0u;
    stackPtr++;

    var iterations = 0u;
    let maxIterations = min(count * 3u, 10000u);

    while (stackPtr > 0u && iterations < maxIterations) {
        iterations++;
        stackPtr--;
        let nodeIdx = stack[stackPtr];

        let node = tlasNodes[nodeIdx];

        var children: array<u32, 4>;
        var dists: array<f32, 4>;

        children[0] = node.child0;
        children[1] = node.child1;
        children[2] = node.child2;
        children[3] = node.child3;

        dists[0] = select(
            intersectAABBDist(ray.origin, invDir,
                vec3(node.c0_minX, node.c0_minY, node.c0_minZ),
                vec3(node.c0_maxX, node.c0_maxY, node.c0_maxZ)),
            AABB_SENTINEL,
            children[0] == INVALID_NODE
        );
        dists[1] = select(
            intersectAABBDist(ray.origin, invDir,
                vec3(node.c1_minX, node.c1_minY, node.c1_minZ),
                vec3(node.c1_maxX, node.c1_maxY, node.c1_maxZ)),
            AABB_SENTINEL,
            children[1] == INVALID_NODE
        );
        dists[2] = select(
            intersectAABBDist(ray.origin, invDir,
                vec3(node.c2_minX, node.c2_minY, node.c2_minZ),
                vec3(node.c2_maxX, node.c2_maxY, node.c2_maxZ)),
            AABB_SENTINEL,
            children[2] == INVALID_NODE
        );
        dists[3] = select(
            intersectAABBDist(ray.origin, invDir,
                vec3(node.c3_minX, node.c3_minY, node.c3_minZ),
                vec3(node.c3_maxX, node.c3_maxY, node.c3_maxZ)),
            AABB_SENTINEL,
            children[3] == INVALID_NODE
        );

        for (var i = 1u; i < 4u; i++) {
            let keyDist = dists[i];
            let keyChild = children[i];
            var j = i;
            while (j > 0u && dists[j - 1u] > keyDist) {
                dists[j] = dists[j - 1u];
                children[j] = children[j - 1u];
                j--;
            }
            dists[j] = keyDist;
            children[j] = keyChild;
        }

        for (var i = 3i; i >= 0i; i--) {
            let child = children[i];
            let dist = dists[i];

            if (child == INVALID_NODE || dist >= closest.t) {
                continue;
            }

            if (isLeaf(child)) {
                let eid = leafIndex(child);
                if (data[eid].baseColor.a <= 0.0) { continue; }
                let prim = getPrimitive(eid);

                let invMatrix = instanceInverses[eid];
                var objRay: Ray;
                objRay.origin = (invMatrix * vec4(ray.origin, 1.0)).xyz;
                objRay.direction = (invMatrix * vec4(ray.direction, 0.0)).xyz;

                let nodeOffset = entityBlasMeta[eid * 4u];
                let triIdOffset = entityBlasMeta[eid * 4u + 1u];
                let triOffset = entityBlasMeta[eid * 4u + 2u];
                let totalTriCount = entityBlasMeta[eid * 4u + 3u];

                var blasHit: HitResult;
                blasHit.hit = false;
                if (totalTriCount > 0u) {
                    blasHit = traceBLAS(objRay, nodeOffset, triIdOffset, triOffset, totalTriCount, closest.t);
                } else if (prim == 0u) {
                    blasHit = intersectUnitBox(objRay, closest.t);
                } else if (prim == 1u) {
                    blasHit = intersectUnitSphere(objRay, closest.t);
                } else if (prim == 2u) {
                    blasHit = intersectUnitCapsule(objRay, closest.t);
                } else if (prim == 3u) {
                    blasHit = intersectUnitPlane(objRay, closest.t);
                }

                if (blasHit.hit && blasHit.t < closest.t) {
                    closest = blasHit;
                    closest.entityId = eid;
                    let normalMat = mat3x3(invMatrix[0].xyz, invMatrix[1].xyz, invMatrix[2].xyz);
                    closest.normal = normalize(transpose(normalMat) * blasHit.normal);
                    closest.worldPos = ray.origin + blasHit.t * ray.direction;
                }
            } else if (stackPtr < MAX_STACK_DEPTH) {
                stack[stackPtr] = child;
                stackPtr++;
            }
        }
    }

    return closest;
}
`;

export const TLAS_BLAS_TRAVERSAL = /* wgsl */ `
${BVH_UTILS_WGSL}

${ANALYTIC_INTERSECTION_WGSL}

${CLOSEST_HIT_FNS}
`;

export const BLAS_SHADOW_WGSL = /* wgsl */ `
fn intersectBLASTriangleShadow(ray: Ray, tri: BLASTriangle) -> f32 {
    let e1 = tri.e1;
    let e2 = tri.e2;
    let h = cross(ray.direction, e2);
    let a = dot(e1, h);
    if (a > -EPSILON && a < EPSILON) { return -1.0; }
    let f = 1.0 / a;
    let s = ray.origin - tri.v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return -1.0; }
    let q = cross(s, e1);
    let v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) { return -1.0; }
    let t = f * dot(e2, q);
    if (t > EPSILON) { return t; }
    return -1.0;
}

fn traceBLASShadow(
    ray: Ray,
    nodeOffset: u32,
    triIdOffset: u32,
    triOffset: u32,
    totalTriCount: u32,
    maxT: f32
) -> bool {
    if (totalTriCount == 0u) { return false; }

    if (totalTriCount == 1u) {
        let triIdx = blasTriIds[triIdOffset];
        let tri = blasTriangles[triOffset + triIdx];
        let t = intersectBLASTriangleShadow(ray, tri);
        return t > 0.0 && t < maxT;
    }

    let invDir = computeInvDir(ray.direction);
    var stack: array<u32, MAX_STACK_DEPTH>;
    var stackPtr = 0u;
    stack[stackPtr] = 0u;
    stackPtr++;

    var iterations = 0u;
    let maxIterations = min(totalTriCount * 3u, 10000u);

    while (stackPtr > 0u && iterations < maxIterations) {
        iterations++;
        stackPtr--;
        let localIdx = stack[stackPtr];
        let node = blasNodes[nodeOffset + localIdx];

        let left = node.leftChild;
        let right = node.rightChild;

        if (isLeaf(left)) {
            let triIdx = blasTriIds[triIdOffset + leafIndex(left)];
            let tri = blasTriangles[triOffset + triIdx];
            let t = intersectBLASTriangleShadow(ray, tri);
            if (t > 0.0 && t < maxT) { return true; }
        } else {
            let leftNode = blasNodes[nodeOffset + left];
            let dist = intersectAABBDist(ray.origin, invDir,
                vec3(leftNode.minX, leftNode.minY, leftNode.minZ),
                vec3(leftNode.maxX, leftNode.maxY, leftNode.maxZ));
            if (dist < maxT && stackPtr < MAX_STACK_DEPTH) {
                stack[stackPtr] = left;
                stackPtr++;
            }
        }

        if (isLeaf(right)) {
            let triIdx = blasTriIds[triIdOffset + leafIndex(right)];
            let tri = blasTriangles[triOffset + triIdx];
            let t = intersectBLASTriangleShadow(ray, tri);
            if (t > 0.0 && t < maxT) { return true; }
        } else {
            let rightNode = blasNodes[nodeOffset + right];
            let dist = intersectAABBDist(ray.origin, invDir,
                vec3(rightNode.minX, rightNode.minY, rightNode.minZ),
                vec3(rightNode.maxX, rightNode.maxY, rightNode.maxZ));
            if (dist < maxT && stackPtr < MAX_STACK_DEPTH) {
                stack[stackPtr] = right;
                stackPtr++;
            }
        }
    }
    return false;
}
`;

export const TLAS_BLAS_SHADOW = /* wgsl */ `
fn traceShadowAnyHit(ray: Ray, tMax: f32) -> bool {
    let count = getInstanceCount();
    if (count == 0u) { return false; }

    let invDir = computeInvDir(ray.direction);

    var stack: array<u32, MAX_STACK_DEPTH>;
    var stackPtr = 0u;
    stack[stackPtr] = 0u;
    stackPtr++;

    var iterations = 0u;
    let maxIterations = min(count * 3u, 10000u);

    while (stackPtr > 0u && iterations < maxIterations) {
        iterations++;
        stackPtr--;
        let nodeIdx = stack[stackPtr];

        let node = tlasNodes[nodeIdx];

        var children: array<u32, 4>;
        var dists: array<f32, 4>;

        children[0] = node.child0;
        children[1] = node.child1;
        children[2] = node.child2;
        children[3] = node.child3;

        dists[0] = select(
            intersectAABBDist(ray.origin, invDir,
                vec3(node.c0_minX, node.c0_minY, node.c0_minZ),
                vec3(node.c0_maxX, node.c0_maxY, node.c0_maxZ)),
            AABB_SENTINEL,
            children[0] == INVALID_NODE
        );
        dists[1] = select(
            intersectAABBDist(ray.origin, invDir,
                vec3(node.c1_minX, node.c1_minY, node.c1_minZ),
                vec3(node.c1_maxX, node.c1_maxY, node.c1_maxZ)),
            AABB_SENTINEL,
            children[1] == INVALID_NODE
        );
        dists[2] = select(
            intersectAABBDist(ray.origin, invDir,
                vec3(node.c2_minX, node.c2_minY, node.c2_minZ),
                vec3(node.c2_maxX, node.c2_maxY, node.c2_maxZ)),
            AABB_SENTINEL,
            children[2] == INVALID_NODE
        );
        dists[3] = select(
            intersectAABBDist(ray.origin, invDir,
                vec3(node.c3_minX, node.c3_minY, node.c3_minZ),
                vec3(node.c3_maxX, node.c3_maxY, node.c3_maxZ)),
            AABB_SENTINEL,
            children[3] == INVALID_NODE
        );

        for (var i = 0u; i < 4u; i++) {
            let child = children[i];
            let dist = dists[i];

            if (child == INVALID_NODE || dist >= tMax) {
                continue;
            }

            if (isLeaf(child)) {
                let eid = leafIndex(child);
                if (data[eid].baseColor.a <= 0.0) { continue; }
                if ((data[eid].flags & 0x1000u) == 0u) { continue; }
                let prim = getPrimitive(eid);

                let invMatrix = instanceInverses[eid];
                var objRay: Ray;
                objRay.origin = (invMatrix * vec4(ray.origin, 1.0)).xyz;
                objRay.direction = (invMatrix * vec4(ray.direction, 0.0)).xyz;

                let nodeOffset = entityBlasMeta[eid * 4u];
                let triIdOffset = entityBlasMeta[eid * 4u + 1u];
                let triOffset = entityBlasMeta[eid * 4u + 2u];
                let totalTriCount = entityBlasMeta[eid * 4u + 3u];

                var anyHit = false;
                if (totalTriCount > 0u) {
                    anyHit = traceBLASShadow(objRay, nodeOffset, triIdOffset, triOffset, totalTriCount, tMax);
                } else if (prim == 0u) {
                    anyHit = intersectUnitBoxShadow(objRay, tMax);
                } else if (prim == 1u) {
                    anyHit = intersectUnitSphereShadow(objRay, tMax);
                } else if (prim == 2u) {
                    anyHit = intersectUnitCapsuleShadow(objRay, tMax);
                } else if (prim == 3u) {
                    anyHit = intersectUnitPlaneShadow(objRay, tMax);
                }

                if (anyHit) {
                    return true;
                }
            } else if (stackPtr < MAX_STACK_DEPTH) {
                stack[stackPtr] = child;
                stackPtr++;
            }
        }
    }

    return false;
}
`;
