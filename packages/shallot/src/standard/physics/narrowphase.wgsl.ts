import {
    solverTypesWGSL,
    narrowphaseBindingsWGSL,
    solverCoreFnsWGSL,
    packKeyWGSL,
    SS_PAIR_TYPE_BASE,
} from "./solver.wgsl";

const sharedNarrowphaseWGSL = /* wgsl */ `
${solverTypesWGSL}
${narrowphaseBindingsWGSL}
${solverCoreFnsWGSL}
${packKeyWGSL}

fn quatConj(q: vec4f) -> vec4f {
    return vec4f(-q.x, -q.y, -q.z, q.w);
}

fn contactCInit(posA: vec3f, rA: vec3f, posB: vec3f, rB: vec3f, n: vec3f, t1: vec3f, t2: vec3f) -> vec3f {
    let cpSep = (posA + rA) - (posB + rB);
    return vec3f(
        dot(n, cpSep) + COLLISION_MARGIN,
        dot(t1, cpSep),
        dot(t2, cpSep),
    );
}

@group(1) @binding(0) var<storage, read_write> pairs: array<u32>;

fn loadWarmstartSearchingHash(bodyA: u32, bodyB: u32, featureKey: u32) -> WarmstartEntry {
    let hCap = params.capacity * params.hashMul;
    for (var s = 0u; s < MAX_PAIR_CONTACTS; s++) {
        let key = packKey(bodyA, bodyB, s);
        let idx = hashLookup(key);
        if (idx < hCap) {
            let ws = warmstarts[idx];
            if (ws.featureKey == featureKey) {
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
    }
    return defaultWarmstart();
}

fn resetWarmstartHash(key: u32) {
    let hCap = params.capacity * params.hashMul;
    let idx = hashLookup(key);
    if (idx < hCap) {
        atomicStore(&solverState[HASH_BASE + idx], HASH_EMPTY);
        warmstarts[idx] = defaultWarmstart();
    }
}

fn pushConstraintSearching(
    bodyA: u32, bodyB: i32, featureKey: u32,
    normal: vec3f, C_init_n: f32,
    tangent1: vec3f, C_init_t1: f32,
    tangent2: vec3f, C_init_t2: f32,
    rA: vec3f, rB: vec3f,
    friction: f32,
    wsKey: u32, wsBodyA: u32, wsBodyB: u32, bilateral: u32,
    fmin_n: f32, fmax_n: f32, cStiffness: f32,
) {
    var bi = bilateral;
    var ct1 = C_init_t1;
    var ct2 = C_init_t2;
    if (bi == CONSTRAINT_CONTACT) {
        let movedA = bodies[bodyA].mass <= 0.0 && bodies[bodyA].moved > 0.5;
        let movedB = bodies[u32(bodyB)].mass <= 0.0 && bodies[u32(bodyB)].moved > 0.5;
        if (movedA || movedB) {
            bi = CONSTRAINT_KINEMATIC;
            if (movedA) {
                ct1 += dot(tangent1, bodies[bodyA].vel);
                ct2 += dot(tangent2, bodies[bodyA].vel);
            }
            if (movedB) {
                ct1 -= dot(tangent1, bodies[u32(bodyB)].vel);
                ct2 -= dot(tangent2, bodies[u32(bodyB)].vel);
            }
        }
    }
    let ws = loadWarmstartSearchingHash(wsBodyA, wsBodyB, featureKey);
    pushConstraintWithWarmstart(bodyA, bodyB, featureKey, normal, C_init_n, tangent1, ct1, tangent2, ct2, rA, rB, friction, wsKey, bi, fmin_n, fmax_n, cStiffness, ws);
}

`;

const emitSingleContactWGSL = /* wgsl */ `
fn emitSingleContact(
    ai: u32, bi: u32,
    normal: vec3f, rA_w: vec3f, rB_w: vec3f,
    posA: vec3f, quatA: vec4f, posB: vec3f, quatB: vec4f,
    fricA: f32, fricB: f32, fkey: u32,
) {
    let lo = min(ai, bi);
    let hi = max(ai, bi);
    let aIsLo = ai < bi;

    let n = select(-normal, normal, aIsLo);
    let tb = tangentBasis(n);
    let mu = sqrt(fricA * fricB);

    let rAl = quatRotate(quatConj(quatA), rA_w);
    let rBl = quatRotate(quatConj(quatB), rB_w);

    let posLo = select(posB, posA, aIsLo);
    let posHi = select(posA, posB, aIsLo);
    let rLo_local = select(rBl, rAl, aIsLo);
    let rHi_local = select(rAl, rBl, aIsLo);
    let rLo_w = select(rB_w, rA_w, aIsLo);
    let rHi_w = select(rA_w, rB_w, aIsLo);

    let ci = contactCInit(posLo, rLo_w, posHi, rHi_w, n, tb[0], tb[1]);

    let wsKey = packKey(lo, hi, 0u);
    pushConstraintSearching(
        lo, i32(hi), fkey,
        n, ci.x,
        tb[0], ci.y,
        tb[1], ci.z,
        rLo_local, rHi_local,
        mu,
        wsKey, lo, hi, 0u,
        -1e30, 0.0, 1e30,
    );
    for (var s = 1u; s < MAX_PAIR_CONTACTS; s++) {
        resetWarmstartHash(packKey(lo, hi, s));
    }
}
`;

const manifoldWGSL = /* wgsl */ `
const MAX_CANDIDATES: u32 = 32u;

struct ManifoldCandidate {
    pointA: vec3f,
    pointB: vec3f,
    depth: f32,
    clipTag: u32,
}

fn reduceManifold(
    candidates: ptr<function, array<ManifoldCandidate, MAX_CANDIDATES>>,
    count: u32,
    normal: vec3f,
    out: ptr<function, array<u32, 4>>,
) -> u32 {
    if (count <= 4u) {
        for (var i = 0u; i < count; i++) { (*out)[i] = i; }
        return count;
    }

    var cx = 0.0; var cy = 0.0; var cz = 0.0;
    for (var i = 0u; i < MAX_CANDIDATES; i++) {
        if (i >= count) { break; }
        cx += (*candidates)[i].pointB.x;
        cy += (*candidates)[i].pointB.y;
        cz += (*candidates)[i].pointB.z;
    }
    let fc = 1.0 / f32(count);
    let center = vec3f(cx * fc, cy * fc, cz * fc);

    let p0rel = (*candidates)[0].pointB - center;
    let p0len = length(p0rel);
    var t0 = select(vec3f(1.0, 0.0, 0.0), p0rel / p0len, p0len > 1e-12);
    let u = normalize(cross(normal, t0));
    let v = cross(normal, u);

    var sel: array<u32, 4> = array(0u, 0u, 0u, 0u);
    var bestProj: array<f32, 4> = array(-1e30, -1e30, -1e30, -1e30);
    for (var i = 0u; i < MAX_CANDIDATES; i++) {
        if (i >= count) { break; }
        let rel = (*candidates)[i].pointB - center;
        let pu = dot(rel, u);
        let pv = dot(rel, v);
        if (pu > bestProj[0]) { bestProj[0] = pu; sel[0] = i; }
        if (-pu > bestProj[1]) { bestProj[1] = -pu; sel[1] = i; }
        if (pv > bestProj[2]) { bestProj[2] = pv; sel[2] = i; }
        if (-pv > bestProj[3]) { bestProj[3] = -pv; sel[3] = i; }
    }

    var unique: array<u32, 4>;
    var uCount = 0u;
    for (var i = 0u; i < 4u; i++) {
        var dup = false;
        for (var j = 0u; j < uCount; j++) {
            if (unique[j] == sel[i]) { dup = true; break; }
        }
        if (!dup) { unique[uCount] = sel[i]; uCount++; }
    }

    var deepIdx = 0u;
    var deepVal = (*candidates)[0].depth;
    for (var i = 1u; i < MAX_CANDIDATES; i++) {
        if (i >= count) { break; }
        if ((*candidates)[i].depth < deepVal) {
            deepVal = (*candidates)[i].depth;
            deepIdx = i;
        }
    }

    var hasDeep = false;
    for (var i = 0u; i < uCount; i++) {
        if (unique[i] == deepIdx) { hasDeep = true; break; }
    }
    if (!hasDeep) {
        if (uCount < 4u) {
            unique[uCount] = deepIdx;
            uCount++;
        } else {
            var shallowest = 0u;
            var shallowestVal = (*candidates)[unique[0]].depth;
            for (var i = 1u; i < uCount; i++) {
                if ((*candidates)[unique[i]].depth > shallowestVal) {
                    shallowestVal = (*candidates)[unique[i]].depth;
                    shallowest = i;
                }
            }
            unique[shallowest] = deepIdx;
        }
    }

    for (var i = 0u; i < uCount; i++) { (*out)[i] = unique[i]; }
    return uCount;
}
`;

const detectBoxBoxWGSL = /* wgsl */ `
${manifoldWGSL}

fn supportPointBox(pos: vec3f, ax0: vec3f, ax1: vec3f, ax2: vec3f, half: vec3f, dir: vec3f) -> vec3f {
    return pos
        + ax0 * select(-half.x, half.x, dot(dir, ax0) >= 0.0)
        + ax1 * select(-half.y, half.y, dot(dir, ax1) >= 0.0)
        + ax2 * select(-half.z, half.z, dot(dir, ax2) >= 0.0);
}

fn detectBoxBox(ci: u32, cj: u32) {
    let bA = bodies[ci];
    let bB = bodies[cj];
    let hA = bA.halfExtents;
    let hB = bB.halfExtents;
    let posA = bA.pos;
    let posB = bB.pos;
    let qA = bA.quat;
    let qB = bB.quat;

    let axA0 = quatRotate(qA, vec3f(1, 0, 0));
    let axA1 = quatRotate(qA, vec3f(0, 1, 0));
    let axA2 = quatRotate(qA, vec3f(0, 0, 1));
    let axB0 = quatRotate(qB, vec3f(1, 0, 0));
    let axB1 = quatRotate(qB, vec3f(0, 1, 0));
    let axB2 = quatRotate(qB, vec3f(0, 0, 1));

    let d = posB - posA;

    var minFacePen = 1e30;
    var bestFaceAxis = vec3f(0.0, 1.0, 0.0);
    var minEdgePen = 1e30;
    var bestEdgeAxis = vec3f(0.0, 1.0, 0.0);
    var separated = false;
    var bestEdgeA = vec3f(0.0);
    var bestEdgeB = vec3f(0.0);
    var bestEdgeIdxA = 0u;
    var bestEdgeIdxB = 0u;

    let faceAxes = array<vec3f, 6>(axA0, axA1, axA2, axB0, axB1, axB2);

    for (var a = 0u; a < 6u; a++) {
        let axis = faceAxes[a];
        let projA = abs(dot(axA0, axis)) * hA.x + abs(dot(axA1, axis)) * hA.y + abs(dot(axA2, axis)) * hA.z;
        let projB = abs(dot(axB0, axis)) * hB.x + abs(dot(axB1, axis)) * hB.y + abs(dot(axB2, axis)) * hB.z;
        let dist_ax = abs(dot(d, axis));
        let pen = projA + projB - dist_ax;
        if (pen < 0.0) { separated = true; break; }
        if (pen < minFacePen) {
            minFacePen = pen;
            bestFaceAxis = axis;
            if (dot(d, axis) < 0.0) { bestFaceAxis = -axis; }
        }
    }

    if (!separated) {
        let edgesA = array<vec3f, 3>(axA0, axA1, axA2);
        let edgesB = array<vec3f, 3>(axB0, axB1, axB2);

        for (var ea = 0u; ea < 3u; ea++) {
            for (var eb = 0u; eb < 3u; eb++) {
                var axis = cross(edgesA[ea], edgesB[eb]);
                let axLen = length(axis);
                if (axLen < 1e-6) { continue; }
                axis /= axLen;

                let projA = abs(dot(axA0, axis)) * hA.x + abs(dot(axA1, axis)) * hA.y + abs(dot(axA2, axis)) * hA.z;
                let projB = abs(dot(axB0, axis)) * hB.x + abs(dot(axB1, axis)) * hB.y + abs(dot(axB2, axis)) * hB.z;
                let dist_ax = abs(dot(d, axis));
                let pen = projA + projB - dist_ax;
                if (pen < 0.0) { separated = true; break; }
                if (pen < minEdgePen) {
                    minEdgePen = pen;
                    bestEdgeAxis = axis;
                    if (dot(d, axis) < 0.0) { bestEdgeAxis = -axis; }
                    bestEdgeA = edgesA[ea];
                    bestEdgeB = edgesB[eb];
                    bestEdgeIdxA = ea;
                    bestEdgeIdxB = eb;
                }
            }
            if (separated) { break; }
        }
    }

    var bestIsFace = true;
    var bestAxis = bestFaceAxis;
    if (!separated && minFacePen < 1e30 && minEdgePen < minFacePen * 0.95 - 0.01) {
        bestIsFace = false;
        bestAxis = bestEdgeAxis;
    }

    if (separated) {
        return;
    }

    var candidates: array<ManifoldCandidate, MAX_CANDIDATES>;
    var candCount = 0u;
    let satNormal = bestAxis;
    let n = bestAxis;

    var faceRefIsA = false;
    var refFaceIdx = 0u;
    var incFaceIdx = 0u;

    if (bestIsFace) {
        let nDotA0 = abs(dot(n, axA0));
        let nDotA1 = abs(dot(n, axA1));
        let nDotA2 = abs(dot(n, axA2));
        let nDotB0 = abs(dot(n, axB0));
        let nDotB1 = abs(dot(n, axB1));
        let nDotB2 = abs(dot(n, axB2));

        var maxDotA = nDotA0;
        if (nDotA1 > maxDotA) { maxDotA = nDotA1; }
        if (nDotA2 > maxDotA) { maxDotA = nDotA2; }
        var maxDotB = nDotB0;
        if (nDotB1 > maxDotB) { maxDotB = nDotB1; }
        if (nDotB2 > maxDotB) { maxDotB = nDotB2; }
        let refIsA = maxDotA >= maxDotB;

        var incVerts: array<vec3f, 4>;
        var refCenter: vec3f;
        var refNormal: vec3f;
        var refTangent1: vec3f;
        var refTangent2: vec3f;
        var refHalf1: f32;
        var refHalf2: f32;

        if (refIsA) {
            faceRefIsA = true;
            refNormal = n;
            if (nDotA0 >= nDotA1 && nDotA0 >= nDotA2) {
                let s = sign(dot(n, axA0));
                refCenter = posA + axA0 * s * hA.x;
                refTangent1 = axA1; refHalf1 = hA.y;
                refTangent2 = axA2; refHalf2 = hA.z;
                refFaceIdx = select(0u, 1u, s > 0.0);
            } else if (nDotA1 >= nDotA2) {
                let s = sign(dot(n, axA1));
                refCenter = posA + axA1 * s * hA.y;
                refTangent1 = axA0; refHalf1 = hA.x;
                refTangent2 = axA2; refHalf2 = hA.z;
                refFaceIdx = 2u + select(0u, 1u, s > 0.0);
            } else {
                let s = sign(dot(n, axA2));
                refCenter = posA + axA2 * s * hA.z;
                refTangent1 = axA0; refHalf1 = hA.x;
                refTangent2 = axA1; refHalf2 = hA.y;
                refFaceIdx = 4u + select(0u, 1u, s > 0.0);
            }

            let negN = -n;
            let dB0 = dot(negN, axB0);
            let dB1 = dot(negN, axB1);
            let dB2 = dot(negN, axB2);
            let aB0 = abs(dB0); let aB1 = abs(dB1); let aB2 = abs(dB2);

            var incAxis: vec3f;
            var incT1: vec3f;
            var incT2: vec3f;
            var incH: f32;
            var incH1: f32;
            var incH2: f32;

            if (aB0 >= aB1 && aB0 >= aB2) {
                let s = sign(dB0);
                incAxis = axB0 * s; incH = hB.x;
                incT1 = axB1; incH1 = hB.y;
                incT2 = axB2; incH2 = hB.z;
                incFaceIdx = select(0u, 1u, s > 0.0);
            } else if (aB1 >= aB2) {
                let s = sign(dB1);
                incAxis = axB1 * s; incH = hB.y;
                incT1 = axB0; incH1 = hB.x;
                incT2 = axB2; incH2 = hB.z;
                incFaceIdx = 2u + select(0u, 1u, s > 0.0);
            } else {
                let s = sign(dB2);
                incAxis = axB2 * s; incH = hB.z;
                incT1 = axB0; incH1 = hB.x;
                incT2 = axB1; incH2 = hB.y;
                incFaceIdx = 4u + select(0u, 1u, s > 0.0);
            }

            let incCenter = posB + incAxis * incH;
            incVerts[0] = incCenter + incT1 * incH1 + incT2 * incH2;
            incVerts[1] = incCenter - incT1 * incH1 + incT2 * incH2;
            incVerts[2] = incCenter - incT1 * incH1 - incT2 * incH2;
            incVerts[3] = incCenter + incT1 * incH1 - incT2 * incH2;
        } else {
            refNormal = -n;
            if (nDotB0 >= nDotB1 && nDotB0 >= nDotB2) {
                let s = sign(dot(-n, axB0));
                refCenter = posB + axB0 * s * hB.x;
                refTangent1 = axB1; refHalf1 = hB.y;
                refTangent2 = axB2; refHalf2 = hB.z;
                refFaceIdx = select(0u, 1u, s > 0.0);
            } else if (nDotB1 >= nDotB2) {
                let s = sign(dot(-n, axB1));
                refCenter = posB + axB1 * s * hB.y;
                refTangent1 = axB0; refHalf1 = hB.x;
                refTangent2 = axB2; refHalf2 = hB.z;
                refFaceIdx = 2u + select(0u, 1u, s > 0.0);
            } else {
                let s = sign(dot(-n, axB2));
                refCenter = posB + axB2 * s * hB.z;
                refTangent1 = axB0; refHalf1 = hB.x;
                refTangent2 = axB1; refHalf2 = hB.y;
                refFaceIdx = 4u + select(0u, 1u, s > 0.0);
            }

            let posN = n;
            let dA0 = dot(posN, axA0);
            let dA1 = dot(posN, axA1);
            let dA2 = dot(posN, axA2);
            let aA0 = abs(dA0); let aA1 = abs(dA1); let aA2 = abs(dA2);

            var incAxis: vec3f;
            var incT1: vec3f;
            var incT2: vec3f;
            var incH: f32;
            var incH1: f32;
            var incH2: f32;

            if (aA0 >= aA1 && aA0 >= aA2) {
                let s = sign(dA0);
                incAxis = axA0 * s; incH = hA.x;
                incT1 = axA1; incH1 = hA.y;
                incT2 = axA2; incH2 = hA.z;
                incFaceIdx = select(0u, 1u, s > 0.0);
            } else if (aA1 >= aA2) {
                let s = sign(dA1);
                incAxis = axA1 * s; incH = hA.y;
                incT1 = axA0; incH1 = hA.x;
                incT2 = axA2; incH2 = hA.z;
                incFaceIdx = 2u + select(0u, 1u, s > 0.0);
            } else {
                let s = sign(dA2);
                incAxis = axA2 * s; incH = hA.z;
                incT1 = axA0; incH1 = hA.x;
                incT2 = axA1; incH2 = hA.y;
                incFaceIdx = 4u + select(0u, 1u, s > 0.0);
            }

            let incCenter = posA + incAxis * incH;
            incVerts[0] = incCenter + incT1 * incH1 + incT2 * incH2;
            incVerts[1] = incCenter - incT1 * incH1 + incT2 * incH2;
            incVerts[2] = incCenter - incT1 * incH1 - incT2 * incH2;
            incVerts[3] = incCenter + incT1 * incH1 - incT2 * incH2;
        }

        var clipIn: array<vec3f, 8>;
        var clipOut: array<vec3f, 8>;
        var clipTags: array<u32, 8>;
        var clipTagsOut: array<u32, 8>;
        var clipCount = 4u;
        for (var v = 0u; v < 4u; v++) { clipIn[v] = incVerts[v]; clipTags[v] = v; }

        let clipNormals = array<vec3f, 4>(refTangent1, -refTangent1, refTangent2, -refTangent2);
        let clipOffsets = array<f32, 4>(
            dot(refTangent1, refCenter) + refHalf1,
            -dot(refTangent1, refCenter) + refHalf1,
            dot(refTangent2, refCenter) + refHalf2,
            -dot(refTangent2, refCenter) + refHalf2,
        );

        for (var p = 0u; p < 4u; p++) {
            let planeN = clipNormals[p];
            let planeD = clipOffsets[p];
            var outCount = 0u;

            var a = clipIn[clipCount - 1u];
            var da = dot(planeN, a) - planeD;
            for (var v = 0u; v < clipCount; v++) {
                let b = clipIn[v];
                let db = dot(planeN, b) - planeD;
                let aInside = da <= 1e-5;
                let bInside = db <= 1e-5;
                if (aInside != bInside) {
                    var t = 0.0;
                    let denom = da - db;
                    if (abs(denom) > 1e-6) { t = clamp(da / denom, 0.0, 1.0); }
                    if (outCount < 8u) { clipTagsOut[outCount] = 4u + p; clipOut[outCount] = a + (b - a) * t; outCount++; }
                }
                if (bInside) {
                    if (outCount < 8u) { clipTagsOut[outCount] = clipTags[v]; clipOut[outCount] = b; outCount++; }
                }
                a = b;
                da = db;
            }

            clipCount = outCount;
            for (var v = 0u; v < clipCount; v++) { clipIn[v] = clipOut[v]; clipTags[v] = clipTagsOut[v]; }
        }

        let refD = dot(refNormal, refCenter);
        for (var v = 0u; v < clipCount; v++) {
            let sep = dot(refNormal, clipIn[v]) - refD;
            if (sep <= 1e-5 && candCount < MAX_CANDIDATES) {
                let pInc = clipIn[v];
                let pRef = pInc - refNormal * sep;
                let cA = select(pInc, pRef, faceRefIsA);
                let cB = select(pRef, pInc, faceRefIsA);
                candidates[candCount] = ManifoldCandidate(cA, cB, sep, clipTags[v]);
                candCount++;
            }
        }

        if (candCount == 0u) {
            let sA = supportPointBox(posA, axA0, axA1, axA2, hA, satNormal);
            let sB = supportPointBox(posB, axB0, axB1, axB2, hB, -satNormal);
            candidates[0] = ManifoldCandidate(sA, sB, dot(sA - sB, satNormal), 0u);
            candCount = 1u;
        }
    } else {
        let eA = bestEdgeA;
        let eB = bestEdgeB;

        let sA0 = dot(axA0, satNormal) > 0.0;
        let sA1 = dot(axA1, satNormal) > 0.0;
        let sA2 = dot(axA2, satNormal) > 0.0;

        var supA = vec3f(0.0);
        supA += axA0 * select(-hA.x, hA.x, sA0);
        supA += axA1 * select(-hA.y, hA.y, sA1);
        supA += axA2 * select(-hA.z, hA.z, sA2);
        let pA = posA + supA;

        let sB0 = dot(axB0, -satNormal) > 0.0;
        let sB1 = dot(axB1, -satNormal) > 0.0;
        let sB2 = dot(axB2, -satNormal) > 0.0;

        var supB = vec3f(0.0);
        supB += axB0 * select(-hB.x, hB.x, sB0);
        supB += axB1 * select(-hB.y, hB.y, sB1);
        supB += axB2 * select(-hB.z, hB.z, sB2);
        let pB = posB + supB;

        var halfLenA = hA.x;
        if (abs(dot(eA, axA1)) > 0.5) { halfLenA = hA.y; }
        if (abs(dot(eA, axA2)) > 0.5) { halfLenA = hA.z; }
        var halfLenB = hB.x;
        if (abs(dot(eB, axB1)) > 0.5) { halfLenB = hB.y; }
        if (abs(dot(eB, axB2)) > 0.5) { halfLenB = hB.z; }

        let dAB = pA - pB;
        let dAe = dot(eA, eA);
        let dBe = dot(eB, eB);
        let dAeB = dot(eA, eB);
        let dAeAB = dot(eA, dAB);
        let dBeAB = dot(eB, dAB);

        let denom = dAe * dBe - dAeB * dAeB;
        let sN = clamp((dAeB * dBeAB - dBe * dAeAB) / max(denom, 1e-12), -halfLenA, halfLenA);
        let tN = clamp((dAe * dBeAB - dAeB * dAeAB) / max(denom, 1e-12), -halfLenB, halfLenB);

        let closestA = pA + eA * sN;
        let closestB = pB + eB * tN;
        let depth = dot(closestA - closestB, satNormal);

        if (depth <= 0.0) {
            candidates[0] = ManifoldCandidate(closestA, closestB, depth, 0u);
            candCount = 1u;
        } else {
            let sA = supportPointBox(posA, axA0, axA1, axA2, hA, satNormal);
            let sB = supportPointBox(posB, axB0, axB1, axB2, hB, -satNormal);
            candidates[0] = ManifoldCandidate(sA, sB, dot(sA - sB, satNormal), 0u);
            candCount = 1u;
        }
    }

    var selected: array<u32, 4>;
    let satCount = reduceManifold(&candidates, candCount, satNormal, &selected);

    let bbBasisN = -satNormal;
    let tb = tangentBasis(bbBasisN);
    let mu = sqrt(bA.friction * bB.friction);

    for (var s = 0u; s < satCount; s++) {
        let ci_s = selected[s];
        let rA_w = candidates[ci_s].pointA - posA;
        let rB_w = candidates[ci_s].pointB - posB;
        let rA = quatRotate(quatConj(bA.quat), rA_w);
        let rB = quatRotate(quatConj(bB.quat), rB_w);
        let bbCI = contactCInit(posA, rA_w, posB, rB_w, bbBasisN, tb[0], tb[1]);

        var newKey = 0u;
        if (bestIsFace) {
            let typeVal = select(1u, 0u, faceRefIsA);
            newKey = (typeVal << 24u) | ((refFaceIdx >> 1u) << 16u) | ((incFaceIdx >> 1u) << 8u) | candidates[ci_s].clipTag;
        } else {
            newKey = (2u << 24u) | ((bestEdgeIdxA & 0xffu) << 8u) | (bestEdgeIdxB & 0xffu);
        }

        let wsKey = packKey(ci, cj, s);
        pushConstraintSearching(
            ci, i32(cj), newKey,
            bbBasisN, bbCI.x,
            tb[0], bbCI.y,
            tb[1], bbCI.z,
            rA, rB,
            mu,
            wsKey, ci, cj, 0u,
            -1e30, 0.0, 1e30,
        );
    }
    for (var s = satCount; s < MAX_PAIR_CONTACTS; s++) {
        resetWarmstartHash(packKey(ci, cj, s));
    }
}
`;

const detectSphereBoxWGSL = /* wgsl */ `
${emitSingleContactWGSL}
fn detectSphereBox(si: u32, bi: u32) {
    let sphere = bodies[si];
    let box = bodies[bi];
    let sPos = sphere.pos;
    let bPos = box.pos;
    let sRadius = sphere.halfExtents.x;
    let h = box.halfExtents;
    let bQ = box.quat;
    let bQc = quatConj(bQ);

    let d = sPos - bPos;
    let local = quatRotate(bQc, d);
    let clamped = clamp(local, -h, h);
    let diff = local - clamped;
    let dist2 = dot(diff, diff);

    let absLocal = abs(local);
    let inside = absLocal.x <= h.x && absLocal.y <= h.y && absLocal.z <= h.z;

    if (!inside && dist2 > 1e-16) {
        let dist = sqrt(dist2);
        let gap = dist - sRadius;
        if (gap > COLLISION_MARGIN) {
            let lo = min(si, bi);
            let hi = max(si, bi);
            for (var s = 0u; s < MAX_PAIR_CONTACTS; s++) {
                resetWarmstartHash(packKey(lo, hi, s));
            }
            return;
        }
        let localNormal = diff / dist;
        let normal = quatRotate(bQ, localNormal);

        let rBox_w = quatRotate(bQ, clamped);
        let rSphere_w = -normal * sRadius;

        emitSingleContact(si, bi, normal, rSphere_w, rBox_w,
            sPos, sphere.quat, bPos, bQ, sphere.friction, box.friction, 4u << 24u);
    } else {
        let face = h - absLocal;
        var minAxis = 0u;
        var minDepth = face.x;
        if (face.y < minDepth) { minAxis = 1u; minDepth = face.y; }
        if (face.z < minDepth) { minAxis = 2u; minDepth = face.z; }

        var localN = vec3f(0.0);
        var cpLocal = local;
        if (minAxis == 0u) {
            let s0 = select(-1.0, 1.0, local.x >= 0.0);
            localN.x = s0;
            cpLocal.x = s0 * h.x;
        } else if (minAxis == 1u) {
            let s0 = select(-1.0, 1.0, local.y >= 0.0);
            localN.y = s0;
            cpLocal.y = s0 * h.y;
        } else {
            let s0 = select(-1.0, 1.0, local.z >= 0.0);
            localN.z = s0;
            cpLocal.z = s0 * h.z;
        }
        let normal = quatRotate(bQ, localN);
        let rBox_w = quatRotate(bQ, cpLocal);
        let rSphere_w = -normal * sRadius;

        emitSingleContact(si, bi, normal, rSphere_w, rBox_w,
            sPos, sphere.quat, bPos, bQ, sphere.friction, box.friction, 4u << 24u);
    }
}
`;

const detectCapsuleBoxWGSL = /* wgsl */ `
fn detectCapsuleBox(ci: u32, bi: u32) {
    let cap = bodies[ci];
    let box = bodies[bi];
    let capAxis = quatRotate(cap.quat, vec3f(0.0, cap.halfExtents.y, 0.0));
    let capR = cap.halfExtents.x;
    let h = box.halfExtents;
    let bQ = box.quat;
    let bQc = quatConj(bQ);

    let lo = min(ci, bi);
    let hi = max(ci, bi);
    let aIsLo = ci < bi;
    let mu = sqrt(cap.friction * box.friction);

    var contactCount = 0u;
    let epA = cap.pos + capAxis;
    let epB = cap.pos - capAxis;

    for (var ep = 0u; ep < 2u; ep++) {
        let sPos = select(epB, epA, ep == 0u);
        let d = sPos - box.pos;
        let local = quatRotate(bQc, d);
        let clamped = clamp(local, -h, h);
        let diff = local - clamped;
        let dist2 = dot(diff, diff);

        let absLocal = abs(local);
        let isInside = absLocal.x <= h.x && absLocal.y <= h.y && absLocal.z <= h.z;

        var normal: vec3f;
        var rBox_w: vec3f;
        var emitThis = false;

        if (!isInside && dist2 > 1e-16) {
            let dist = sqrt(dist2);
            let gap = dist - capR;
            if (gap <= COLLISION_MARGIN) {
                normal = quatRotate(bQ, diff / dist);
                rBox_w = quatRotate(bQ, clamped);
                emitThis = true;
            }
        } else {
            let face = h - absLocal;
            var minAxis = 0u;
            var minVal = face.x;
            if (face.y < minVal) { minAxis = 1u; minVal = face.y; }
            if (face.z < minVal) { minAxis = 2u; minVal = face.z; }
            var localN = vec3f(0.0);
            var cpLocal = local;
            if (minAxis == 0u) {
                let s0 = select(-1.0, 1.0, local.x >= 0.0);
                localN.x = s0; cpLocal.x = s0 * h.x;
            } else if (minAxis == 1u) {
                let s0 = select(-1.0, 1.0, local.y >= 0.0);
                localN.y = s0; cpLocal.y = s0 * h.y;
            } else {
                let s0 = select(-1.0, 1.0, local.z >= 0.0);
                localN.z = s0; cpLocal.z = s0 * h.z;
            }
            normal = quatRotate(bQ, localN);
            rBox_w = quatRotate(bQ, cpLocal);
            emitThis = true;
        }

        if (emitThis) {
            let rCap_w = (sPos - cap.pos) + (-normal * capR);
            let n = select(-normal, normal, aIsLo);
            let tb = tangentBasis(n);

            let rCapL = quatRotate(quatConj(cap.quat), rCap_w);
            let rBoxL = quatRotate(bQc, rBox_w);

            let posLo = select(box.pos, cap.pos, aIsLo);
            let posHi = select(cap.pos, box.pos, aIsLo);
            let rLo_l = select(rBoxL, rCapL, aIsLo);
            let rHi_l = select(rCapL, rBoxL, aIsLo);
            let rLo_w = select(rBox_w, rCap_w, aIsLo);
            let rHi_w = select(rCap_w, rBox_w, aIsLo);

            let cInit = contactCInit(posLo, rLo_w, posHi, rHi_w, n, tb[0], tb[1]);
            let fkey = (7u << 24u) | ep;

            pushConstraintSearching(
                lo, i32(hi), fkey,
                n, cInit.x,
                tb[0], cInit.y,
                tb[1], cInit.z,
                rLo_l, rHi_l,
                mu,
                packKey(lo, hi, contactCount), lo, hi, 0u,
                -1e30, 0.0, 1e30,
            );
            contactCount++;
        }
    }

    for (var s = contactCount; s < MAX_PAIR_CONTACTS; s++) {
        resetWarmstartHash(packKey(lo, hi, s));
    }
}
`;

const detectSphereSphereWGSL = /* wgsl */ `
fn detectSphereSphere(ci: u32, cj: u32) {
    let bA = bodies[ci];
    let bB = bodies[cj];
    let posA = bA.pos;
    let posB = bB.pos;
    let rA = bA.halfExtents.x;
    let rB = bB.halfExtents.x;

    let d = posA - posB;
    let dist = length(d);
    let gap = dist - rA - rB;
    if (gap > COLLISION_MARGIN) {
        for (var s = 0u; s < MAX_PAIR_CONTACTS; s++) {
            resetWarmstartHash(packKey(ci, cj, s));
        }
        return;
    }

    var normal: vec3f;
    if (dist < 1e-8) {
        normal = vec3f(0.0, 1.0, 0.0);
    } else {
        normal = d / dist;
    }

    let tb = tangentBasis(normal);
    let mu = sqrt(bA.friction * bB.friction);

    let rA_w = -normal * rA;
    let rB_w = normal * rB;
    let rA_local = quatRotate(quatConj(bA.quat), rA_w);
    let rB_local = quatRotate(quatConj(bB.quat), rB_w);
    let ssCI = contactCInit(posA, rA_w, posB, rB_w, normal, tb[0], tb[1]);

    let featureKey = 3u << 24u;
    let wsKey = packKey(ci, cj, 0u);
    pushConstraintSearching(
        ci, i32(cj), featureKey,
        normal, ssCI.x,
        tb[0], ssCI.y,
        tb[1], ssCI.z,
        rA_local, rB_local,
        mu,
        wsKey, ci, cj, 0u,
        -1e30, 0.0, 1e30,
    );
    for (var s = 1u; s < MAX_PAIR_CONTACTS; s++) {
        resetWarmstartHash(packKey(ci, cj, s));
    }
}
`;

const closestPointOnSegmentWGSL = /* wgsl */ `
fn closestPointOnSegment(p: vec3f, a: vec3f, b: vec3f) -> vec3f {
    let ab = b - a;
    let ab2 = dot(ab, ab);
    if (ab2 < 1e-12) { return a; }
    let t = clamp(dot(p - a, ab) / ab2, 0.0, 1.0);
    return a + ab * t;
}
`;

const closestPointsOnSegmentsWGSL = /* wgsl */ `
fn closestPointsOnSegments(p0: vec3f, p1: vec3f, q0: vec3f, q1: vec3f) -> array<vec3f, 2> {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    var s = 0.0;
    var t = 0.0;

    if (a <= 1e-12 && e <= 1e-12) {
        return array(p0, q0);
    }

    if (a <= 1e-12) {
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e <= 1e-12) {
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;

            if (abs(denom) > 1e-12) {
                s = clamp((b * f - c * e) / denom, 0.0, 1.0);
            }

            t = (b * s + f) / e;

            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }

    return array(p0 + d1 * s, q0 + d2 * t);
}
`;

const detectCapsuleSphereWGSL = /* wgsl */ `
${closestPointOnSegmentWGSL}
${emitSingleContactWGSL}

fn detectCapsuleSphere(ci: u32, si: u32) {
    let cap = bodies[ci];
    let sph = bodies[si];
    let capAxis = quatRotate(cap.quat, vec3f(0.0, cap.halfExtents.y, 0.0));
    let segA = cap.pos + capAxis;
    let segB = cap.pos - capAxis;
    let capR = cap.halfExtents.x;
    let sphR = sph.halfExtents.x;

    let closest = closestPointOnSegment(sph.pos, segA, segB);
    let d = closest - sph.pos;
    let dist = length(d);
    let gap = dist - capR - sphR;

    let lo = min(ci, si);
    let hi = max(ci, si);
    if (gap > COLLISION_MARGIN) {
        for (var s = 0u; s < MAX_PAIR_CONTACTS; s++) {
            resetWarmstartHash(packKey(lo, hi, s));
        }
        return;
    }

    var normal: vec3f;
    if (dist < 1e-8) {
        normal = vec3f(0.0, 1.0, 0.0);
    } else {
        normal = d / dist;
    }

    let rCap_w = (closest - cap.pos) + (-normal * capR);
    let rSph_w = normal * sphR;

    emitSingleContact(ci, si, normal, rCap_w, rSph_w,
        cap.pos, cap.quat, sph.pos, sph.quat,
        cap.friction, sph.friction, 5u << 24u);
}
`;

const detectCapsuleCapsuleWGSL = /* wgsl */ `
${closestPointsOnSegmentsWGSL}
${emitSingleContactWGSL}

fn detectCapsuleCapsule(ci: u32, cj: u32) {
    let bA = bodies[ci];
    let bB = bodies[cj];
    let axisA = quatRotate(bA.quat, vec3f(0.0, bA.halfExtents.y, 0.0));
    let axisB = quatRotate(bB.quat, vec3f(0.0, bB.halfExtents.y, 0.0));
    let rA = bA.halfExtents.x;
    let rB = bB.halfExtents.x;

    let cp = closestPointsOnSegments(
        bA.pos + axisA, bA.pos - axisA,
        bB.pos + axisB, bB.pos - axisB);
    let d = cp[0] - cp[1];
    let dist = length(d);
    let gap = dist - rA - rB;

    if (gap > COLLISION_MARGIN) {
        for (var s = 0u; s < MAX_PAIR_CONTACTS; s++) {
            resetWarmstartHash(packKey(ci, cj, s));
        }
        return;
    }

    var normal: vec3f;
    if (dist < 1e-8) {
        normal = vec3f(0.0, 1.0, 0.0);
    } else {
        normal = d / dist;
    }

    let rA_w = (cp[0] - bA.pos) + (-normal * rA);
    let rB_w = (cp[1] - bB.pos) + (normal * rB);

    emitSingleContact(ci, cj, normal, rA_w, rB_w,
        bA.pos, bA.quat, bB.pos, bB.quat,
        bA.friction, bB.friction, 6u << 24u);
}
`;

const hullAccessWGSL = /* wgsl */ `
${manifoldWGSL}
@group(1) @binding(1) var<storage, read> hullData: array<u32>;

const MAX_HULL_VERTS: u32 = 64u;
const MAX_HULL_FACES: u32 = 32u;
const MAX_HULL_EDGES: u32 = 48u;
const MAX_FACE_VERTS: u32 = 16u;
const MAX_CLIP_VERTS: u32 = 32u;

struct HullMeta {
    vertexBase: u32,
    vertexCount: u32,
    faceBase: u32,
    faceCount: u32,
    edgeBase: u32,
    edgeCount: u32,
    invExtent: vec3f,
}

fn loadHullMeta(hullId: u32) -> HullMeta {
    let b = hullId * 12u;
    return HullMeta(
        hullData[b], hullData[b+1u], hullData[b+2u], hullData[b+3u],
        hullData[b+4u], hullData[b+5u],
        vec3f(bitcast<f32>(hullData[b+6u]), bitcast<f32>(hullData[b+7u]), bitcast<f32>(hullData[b+8u])),
    );
}

fn hullScale(hm: HullMeta, halfExt: vec3f) -> vec3f {
    return halfExt * hm.invExtent;
}

fn hullVertex(hm: HullMeta, idx: u32) -> vec3f {
    let b = hm.vertexBase + idx * 4u;
    return vec3f(bitcast<f32>(hullData[b]), bitcast<f32>(hullData[b+1u]), bitcast<f32>(hullData[b+2u]));
}

fn hullFacePlane(hm: HullMeta, faceIdx: u32, scale: vec3f) -> vec4f {
    let b = hm.faceBase + faceIdx * 8u;
    let rawN = vec3f(bitcast<f32>(hullData[b]), bitcast<f32>(hullData[b+1u]), bitcast<f32>(hullData[b+2u]));
    let rawD = bitcast<f32>(hullData[b+3u]);
    let sn = rawN / scale;
    let snLen = length(sn);
    return vec4f(sn / snLen, rawD / snLen);
}

fn hullFaceIdxBase(hm: HullMeta, faceIdx: u32) -> u32 {
    return hullData[hm.faceBase + faceIdx * 8u + 4u];
}

fn hullFaceIdxCount(hm: HullMeta, faceIdx: u32) -> u32 {
    return hullData[hm.faceBase + faceIdx * 8u + 5u];
}

fn hullFaceVertIdx(base: u32, i: u32) -> u32 {
    return hullData[base + i];
}

fn hullEdge(hm: HullMeta, idx: u32, scale: vec3f) -> vec3f {
    let b = hm.edgeBase + idx * 4u;
    let raw = vec3f(bitcast<f32>(hullData[b]), bitcast<f32>(hullData[b+1u]), bitcast<f32>(hullData[b+2u]));
    return raw * scale;
}

fn projectHullOnAxis(hm: HullMeta, pos: vec3f, quat: vec4f, axis: vec3f, scale: vec3f) -> vec2f {
    var mn = 1e30;
    var mx = -1e30;
    for (var i = 0u; i < MAX_HULL_VERTS; i++) {
        if (i >= hm.vertexCount) { break; }
        let wv = pos + quatRotate(quat, hullVertex(hm, i) * scale);
        let d = dot(wv, axis);
        mn = min(mn, d);
        mx = max(mx, d);
    }
    return vec2f(mn, mx);
}

fn closestPointOnHull(hm: HullMeta, hQc: vec4f, worldOffset: vec3f, sRadius: f32, scale: vec3f) -> vec4f {
    let localCenter = quatRotate(hQc, worldOffset);
    let scaledCenter = localCenter / scale;
    var closestDist = 1e30;
    var closestPoint = vec3f(0.0);

    for (var fi = 0u; fi < MAX_HULL_FACES; fi++) {
        if (fi >= hm.faceCount) { break; }
        let b = hm.faceBase + fi * 8u;
        let fn0 = vec3f(bitcast<f32>(hullData[b]), bitcast<f32>(hullData[b+1u]), bitcast<f32>(hullData[b+2u]));
        let fd = bitcast<f32>(hullData[b+3u]);
        let dist = dot(fn0, scaledCenter) + fd;
        let scaledRadius = sRadius / min(scale.x, min(scale.y, scale.z));
        if (dist < -scaledRadius) { continue; }
        let projected = scaledCenter - fn0 * dist;
        let idxBase = hullFaceIdxBase(hm, fi);
        let idxCount = hullFaceIdxCount(hm, fi);
        var inside = true;
        for (var ei = 0u; ei < MAX_FACE_VERTS; ei++) {
            if (ei >= idxCount) { break; }
            let va = hullVertex(hm, hullFaceVertIdx(idxBase, ei));
            let vb = hullVertex(hm, hullFaceVertIdx(idxBase, (ei + 1u) % idxCount));
            if (dot(cross(vb - va, projected - va), fn0) < -1e-5) { inside = false; break; }
        }
        if (inside) {
            let absDist = abs(dist);
            if (absDist < closestDist) { closestDist = absDist; closestPoint = projected; }
        }
    }

    for (var fi = 0u; fi < MAX_HULL_FACES; fi++) {
        if (fi >= hm.faceCount) { break; }
        let idxBase = hullFaceIdxBase(hm, fi);
        let idxCount = hullFaceIdxCount(hm, fi);
        for (var ei = 0u; ei < MAX_FACE_VERTS; ei++) {
            if (ei >= idxCount) { break; }
            let va = hullVertex(hm, hullFaceVertIdx(idxBase, ei));
            let vb = hullVertex(hm, hullFaceVertIdx(idxBase, (ei + 1u) % idxCount));
            let ab = vb - va;
            let ab2 = dot(ab, ab);
            var cp = va;
            if (ab2 > 1e-12) { cp = va + ab * clamp(dot(scaledCenter - va, ab) / ab2, 0.0, 1.0); }
            let dist = length(scaledCenter - cp);
            if (dist < closestDist) { closestDist = dist; closestPoint = cp; }
        }
    }

    for (var vi = 0u; vi < MAX_HULL_VERTS; vi++) {
        if (vi >= hm.vertexCount) { break; }
        let v = hullVertex(hm, vi);
        let dist = length(scaledCenter - v);
        if (dist < closestDist) { closestDist = dist; closestPoint = v; }
    }

    let worldPoint = closestPoint * scale;
    let worldDist = length(localCenter - worldPoint);
    return vec4f(worldPoint, worldDist - sRadius);
}
`;

const detectHullBoxWGSL = /* wgsl */ `
${hullAccessWGSL}
${emitSingleContactWGSL}

fn detectHullBox(hui: u32, bi: u32) {
    let hBody = bodies[hui];
    let bBody = bodies[bi];
    let hm = loadHullMeta(hBody.hullId);
    let hPos = hBody.pos;
    let hQ = hBody.quat;
    let bPos = bBody.pos;
    let bQ = bBody.quat;
    let hB = bBody.halfExtents;
    let S = hullScale(hm, hBody.halfExtents);

    let axB0 = quatRotate(bQ, vec3f(1, 0, 0));
    let axB1 = quatRotate(bQ, vec3f(0, 1, 0));
    let axB2 = quatRotate(bQ, vec3f(0, 0, 1));
    let d = bPos - hPos;

    var minPen = 1e30;
    var bestAxis = vec3f(0.0, 1.0, 0.0);
    var separated = false;

    // Hull face normals
    for (var fi = 0u; fi < MAX_HULL_FACES; fi++) {
        if (fi >= hm.faceCount) { break; }
        let plane = hullFacePlane(hm, fi, S);
        let axis = quatRotate(hQ, plane.xyz);
        let hProj = projectHullOnAxis(hm, hPos, hQ, axis, S);
        let bProj = abs(dot(axB0, axis)) * hB.x + abs(dot(axB1, axis)) * hB.y + abs(dot(axB2, axis)) * hB.z;
        let bCenter = dot(bPos, axis);
        let pen = min(hProj.y - (bCenter - bProj), (bCenter + bProj) - hProj.x);
        if (pen < 0.0) { separated = true; break; }
        if (pen < minPen * 0.95 - 0.01) {
            minPen = pen;
            bestAxis = axis;
            if (dot(d, axis) < 0.0) { bestAxis = -axis; }
        }
    }

    // Box face normals
    if (!separated) {
        let boxAxes = array<vec3f, 3>(axB0, axB1, axB2);
        for (var a = 0u; a < 3u; a++) {
            let axis = boxAxes[a];
            let hProj = projectHullOnAxis(hm, hPos, hQ, axis, S);
            let bProj = abs(dot(axB0, axis)) * hB.x + abs(dot(axB1, axis)) * hB.y + abs(dot(axB2, axis)) * hB.z;
            let bCenter = dot(bPos, axis);
            let pen = min(hProj.y - (bCenter - bProj), (bCenter + bProj) - hProj.x);
            if (pen < 0.0) { separated = true; break; }
            if (pen < minPen * 0.95 - 0.01) {
                minPen = pen;
                bestAxis = axis;
                if (dot(d, axis) < 0.0) { bestAxis = -axis; }
            }
        }
    }

    // Edge-edge cross products
    if (!separated) {
        let boxEdges = array<vec3f, 3>(axB0, axB1, axB2);
        for (var ea = 0u; ea < MAX_HULL_EDGES; ea++) {
            if (ea >= hm.edgeCount) { break; }
            let edgeA = quatRotate(hQ, hullEdge(hm, ea, S));
            for (var eb = 0u; eb < 3u; eb++) {
                var axis = cross(edgeA, boxEdges[eb]);
                let axLen = length(axis);
                if (axLen < 1e-6) { continue; }
                axis /= axLen;
                let hProj = projectHullOnAxis(hm, hPos, hQ, axis, S);
                let bProj = abs(dot(axB0, axis)) * hB.x + abs(dot(axB1, axis)) * hB.y + abs(dot(axB2, axis)) * hB.z;
                let bCenter = dot(bPos, axis);
                let pen = min(hProj.y - (bCenter - bProj), (bCenter + bProj) - hProj.x);
                if (pen < 0.0) { separated = true; break; }
                if (pen < minPen * 0.95 - 0.01) {
                    minPen = pen;
                    bestAxis = axis;
                    if (dot(d, axis) < 0.0) { bestAxis = -axis; }
                }
            }
            if (separated) { break; }
        }
    }

    if (separated) { return; }

    let n = bestAxis;

    // Reference face: on hull, most aligned with n
    var refFaceIdx = 0u;
    var refDmax = -1e30;
    for (var fi = 0u; fi < MAX_HULL_FACES; fi++) {
        if (fi >= hm.faceCount) { break; }
        let fn0 = quatRotate(hQ, hullFacePlane(hm, fi, S).xyz);
        let dd = dot(fn0, n);
        if (dd > refDmax) { refDmax = dd; refFaceIdx = fi; }
    }

    let refPlane = hullFacePlane(hm, refFaceIdx, S);
    let refNormal = quatRotate(hQ, refPlane.xyz);
    let refIdxBase = hullFaceIdxBase(hm, refFaceIdx);
    let refIdxCount = hullFaceIdxCount(hm, refFaceIdx);

    // Incident face: box face most anti-aligned with n
    let negN = -n;
    let dB0 = dot(negN, axB0); let dB1 = dot(negN, axB1); let dB2 = dot(negN, axB2);
    let aB0 = abs(dB0); let aB1 = abs(dB1); let aB2 = abs(dB2);
    var incVerts: array<vec3f, 4>;
    var incFaceIdx = 0u;
    if (aB0 >= aB1 && aB0 >= aB2) {
        let s = sign(dB0);
        let c0 = bPos + axB0 * s * hB.x;
        incVerts[0] = c0 + axB1 * hB.y + axB2 * hB.z;
        incVerts[1] = c0 - axB1 * hB.y + axB2 * hB.z;
        incVerts[2] = c0 - axB1 * hB.y - axB2 * hB.z;
        incVerts[3] = c0 + axB1 * hB.y - axB2 * hB.z;
        incFaceIdx = select(0u, 1u, s > 0.0);
    } else if (aB1 >= aB2) {
        let s = sign(dB1);
        let c0 = bPos + axB1 * s * hB.y;
        incVerts[0] = c0 + axB0 * hB.x + axB2 * hB.z;
        incVerts[1] = c0 - axB0 * hB.x + axB2 * hB.z;
        incVerts[2] = c0 - axB0 * hB.x - axB2 * hB.z;
        incVerts[3] = c0 + axB0 * hB.x - axB2 * hB.z;
        incFaceIdx = 2u + select(0u, 1u, s > 0.0);
    } else {
        let s = sign(dB2);
        let c0 = bPos + axB2 * s * hB.z;
        incVerts[0] = c0 + axB0 * hB.x + axB1 * hB.y;
        incVerts[1] = c0 - axB0 * hB.x + axB1 * hB.y;
        incVerts[2] = c0 - axB0 * hB.x - axB1 * hB.y;
        incVerts[3] = c0 + axB0 * hB.x - axB1 * hB.y;
        incFaceIdx = 4u + select(0u, 1u, s > 0.0);
    }

    // Clip incident face against reference face edge planes
    var clipIn: array<vec3f, MAX_CLIP_VERTS>;
    var clipOut: array<vec3f, MAX_CLIP_VERTS>;
    var clipCount = 4u;
    for (var v = 0u; v < 4u; v++) { clipIn[v] = incVerts[v]; }

    for (var ei = 0u; ei < MAX_FACE_VERTS; ei++) {
        if (ei >= refIdxCount) { break; }
        let vi = hullFaceVertIdx(refIdxBase, ei);
        let vj = hullFaceVertIdx(refIdxBase, (ei + 1u) % refIdxCount);
        let va = hPos + quatRotate(hQ, hullVertex(hm, vi) * S);
        let vb = hPos + quatRotate(hQ, hullVertex(hm, vj) * S);
        let edge0 = va - vb;
        let planeN = -cross(edge0, refNormal);
        let planeD = dot(va, planeN);
        var outCount = 0u;
        var a = clipIn[clipCount - 1u];
        var da = dot(planeN, a) - planeD;
        for (var v = 0u; v < MAX_CLIP_VERTS; v++) {
            if (v >= clipCount) { break; }
            let b = clipIn[v];
            let db = dot(planeN, b) - planeD;
            if ((da <= 1e-5) != (db <= 1e-5)) {
                var t = 0.0;
                let denom = da - db;
                if (abs(denom) > 1e-6) { t = clamp(da / denom, 0.0, 1.0); }
                if (outCount < MAX_CLIP_VERTS) { clipOut[outCount] = a + (b - a) * t; outCount++; }
            }
            if (db <= 1e-5) {
                if (outCount < MAX_CLIP_VERTS) { clipOut[outCount] = b; outCount++; }
            }
            a = b;
            da = db;
        }
        clipCount = outCount;
        for (var v = 0u; v < MAX_CLIP_VERTS; v++) {
            if (v >= clipCount) { break; }
            clipIn[v] = clipOut[v];
        }
    }

    // Collect all candidates behind reference face
    let localPlaneEq = refPlane.w;
    let worldPlaneEq = localPlaneEq - dot(refNormal, hPos);
    var candidates: array<ManifoldCandidate, MAX_CANDIDATES>;
    var candCount = 0u;

    for (var v = 0u; v < MAX_CLIP_VERTS; v++) {
        if (v >= clipCount) { break; }
        let depth = dot(refNormal, clipIn[v]) + worldPlaneEq;
        if (depth <= 0.0 && candCount < MAX_CANDIDATES) {
            let pB = clipIn[v];
            let pA = pB - refNormal * depth;
            candidates[candCount] = ManifoldCandidate(pA, pB, depth, v);
            candCount++;
        }
    }

    // Reduce to 4 well-distributed contacts
    var selected: array<u32, 4>;
    let satCount = reduceManifold(&candidates, candCount, n, &selected);

    let bbBasisN = -n;
    let tb = tangentBasis(bbBasisN);
    let mu = sqrt(hBody.friction * bBody.friction);

    for (var s = 0u; s < satCount; s++) {
        let c = candidates[selected[s]];
        let rH_w = c.pointA - hPos;
        let rB_w = c.pointB - bPos;
        let rH = quatRotate(quatConj(hQ), rH_w);
        let rB = quatRotate(quatConj(bQ), rB_w);
        let bbCI = contactCInit(hPos, rH_w, bPos, rB_w, bbBasisN, tb[0], tb[1]);
        let fkey = (10u << 24u) | s;

        let wsKey = packKey(hui, bi, s);
        pushConstraintSearching(
            hui, i32(bi), fkey,
            bbBasisN, bbCI.x,
            tb[0], bbCI.y,
            tb[1], bbCI.z,
            rH, rB,
            mu,
            wsKey, hui, bi, 0u,
            -1e30, 0.0, 1e30,
        );
    }
    for (var s = satCount; s < MAX_PAIR_CONTACTS; s++) {
        resetWarmstartHash(packKey(hui, bi, s));
    }
}
`;

const detectHullSphereWGSL = /* wgsl */ `
${hullAccessWGSL}
${emitSingleContactWGSL}

fn detectHullSphere(hui: u32, si: u32) {
    let hBody = bodies[hui];
    let sBody = bodies[si];
    let hm = loadHullMeta(hBody.hullId);
    let hPos = hBody.pos;
    let hQ = hBody.quat;
    let hQc = quatConj(hQ);
    let sPos = sBody.pos;
    let sRadius = sBody.halfExtents.x;
    let S = hullScale(hm, hBody.halfExtents);

    let result = closestPointOnHull(hm, hQc, sPos - hPos, sRadius, S);
    let penetration = result.w;
    if (penetration > COLLISION_MARGIN) {
        for (var s = 0u; s < MAX_PAIR_CONTACTS; s++) {
            resetWarmstartHash(packKey(min(hui, si), max(hui, si), s));
        }
        return;
    }

    let closestWorld = hPos + quatRotate(hQ, result.xyz);
    let diff = sPos - closestWorld;
    let diffLen = length(diff);
    var normal: vec3f;
    if (diffLen < 1e-8) {
        normal = vec3f(0.0, 1.0, 0.0);
    } else {
        normal = diff / diffLen;
    }

    let rH_w = closestWorld - hPos;
    let rS_w = -normal * sRadius;

    emitSingleContact(si, hui, normal, rS_w, rH_w,
        sPos, sBody.quat, hPos, hQ,
        sBody.friction, hBody.friction, 8u << 24u);
}
`;

const detectHullCapsuleWGSL = /* wgsl */ `
${hullAccessWGSL}

fn detectHullCapsule(hui: u32, ci: u32) {
    let hBody = bodies[hui];
    let cBody = bodies[ci];
    let hm = loadHullMeta(hBody.hullId);
    let hPos = hBody.pos;
    let hQ = hBody.quat;
    let hQc = quatConj(hQ);
    let capAxis = quatRotate(cBody.quat, vec3f(0.0, cBody.halfExtents.y, 0.0));
    let capR = cBody.halfExtents.x;
    let S = hullScale(hm, hBody.halfExtents);

    let lo = min(hui, ci);
    let hi = max(hui, ci);
    let aIsLo = hui < ci;
    let mu = sqrt(hBody.friction * cBody.friction);

    var contactCount = 0u;
    let epA = cBody.pos + capAxis;
    let epB = cBody.pos - capAxis;

    for (var ep = 0u; ep < 2u; ep++) {
        let sPos = select(epB, epA, ep == 0u);
        let result = closestPointOnHull(hm, hQc, sPos - hPos, capR, S);
        let penetration = result.w;
        if (penetration > COLLISION_MARGIN) { continue; }

        let closestWorld = hPos + quatRotate(hQ, result.xyz);
        let diff = sPos - closestWorld;
        let diffLen = length(diff);
        var normal: vec3f;
        if (diffLen < 1e-8) { normal = vec3f(0.0, 1.0, 0.0); }
        else { normal = diff / diffLen; }

        let rH_w = closestWorld - hPos;
        let rC_w = (sPos - cBody.pos) + (-normal * capR);
        let n = select(normal, -normal, aIsLo);
        let tb = tangentBasis(n);
        let rHL = quatRotate(hQc, rH_w);
        let rCL = quatRotate(quatConj(cBody.quat), rC_w);
        let posLo = select(cBody.pos, hPos, aIsLo);
        let posHi = select(hPos, cBody.pos, aIsLo);
        let rLo_l = select(rCL, rHL, aIsLo);
        let rHi_l = select(rHL, rCL, aIsLo);
        let rLo_w = select(rC_w, rH_w, aIsLo);
        let rHi_w = select(rH_w, rC_w, aIsLo);
        let cInit = contactCInit(posLo, rLo_w, posHi, rHi_w, n, tb[0], tb[1]);
        let fkey = (9u << 24u) | ep;

        pushConstraintSearching(
            lo, i32(hi), fkey,
            n, cInit.x,
            tb[0], cInit.y,
            tb[1], cInit.z,
            rLo_l, rHi_l,
            mu,
            packKey(lo, hi, contactCount), lo, hi, 0u,
            -1e30, 0.0, 1e30,
        );
        contactCount++;
    }
    for (var s = contactCount; s < MAX_PAIR_CONTACTS; s++) {
        resetWarmstartHash(packKey(lo, hi, s));
    }
}
`;

const detectHullHullWGSL = /* wgsl */ `
${hullAccessWGSL}

fn detectHullHull(ai: u32, bi: u32) {
    let bA = bodies[ai];
    let bB = bodies[bi];
    let metaA = loadHullMeta(bA.hullId);
    let metaB = loadHullMeta(bB.hullId);
    let posA = bA.pos;
    let posB = bB.pos;
    let qA = bA.quat;
    let qB = bB.quat;
    let d = posB - posA;
    let sA = hullScale(metaA, bA.halfExtents);
    let sB = hullScale(metaB, bB.halfExtents);

    var minPen = 1e30;
    var bestAxis = vec3f(0.0, 1.0, 0.0);
    var separated = false;

    // Face normals from A
    for (var fi = 0u; fi < MAX_HULL_FACES; fi++) {
        if (fi >= metaA.faceCount) { break; }
        let axis = quatRotate(qA, hullFacePlane(metaA, fi, sA).xyz);
        let pA = projectHullOnAxis(metaA, posA, qA, axis, sA);
        let pB = projectHullOnAxis(metaB, posB, qB, axis, sB);
        let pen = min(pA.y - pB.x, pB.y - pA.x);
        if (pen < 0.0) { separated = true; break; }
        if (pen < minPen * 0.95 - 0.01) {
            minPen = pen;
            bestAxis = axis;
            if (dot(d, axis) < 0.0) { bestAxis = -axis; }
        }
    }

    // Face normals from B
    if (!separated) {
        for (var fi = 0u; fi < MAX_HULL_FACES; fi++) {
            if (fi >= metaB.faceCount) { break; }
            let axis = quatRotate(qB, hullFacePlane(metaB, fi, sB).xyz);
            let pA = projectHullOnAxis(metaA, posA, qA, axis, sA);
            let pB = projectHullOnAxis(metaB, posB, qB, axis, sB);
            let pen = min(pA.y - pB.x, pB.y - pA.x);
            if (pen < 0.0) { separated = true; break; }
            if (pen < minPen * 0.95 - 0.01) {
                minPen = pen;
                bestAxis = axis;
                if (dot(d, axis) < 0.0) { bestAxis = -axis; }
            }
        }
    }

    // Edge-edge cross products
    if (!separated) {
        for (var ea = 0u; ea < MAX_HULL_EDGES; ea++) {
            if (ea >= metaA.edgeCount) { break; }
            let edgeA = quatRotate(qA, hullEdge(metaA, ea, sA));
            for (var eb = 0u; eb < MAX_HULL_EDGES; eb++) {
                if (eb >= metaB.edgeCount) { break; }
                let edgeB = quatRotate(qB, hullEdge(metaB, eb, sB));
                var axis = cross(edgeA, edgeB);
                let axLen = length(axis);
                if (axLen < 1e-6) { continue; }
                axis /= axLen;
                let pA = projectHullOnAxis(metaA, posA, qA, axis, sA);
                let pB = projectHullOnAxis(metaB, posB, qB, axis, sB);
                let pen = min(pA.y - pB.x, pB.y - pA.x);
                if (pen < 0.0) { separated = true; break; }
                if (pen < minPen * 0.95 - 0.01) {
                    minPen = pen;
                    bestAxis = axis;
                    if (dot(d, axis) < 0.0) { bestAxis = -axis; }
                }
            }
            if (separated) { break; }
        }
    }

    if (separated) { return; }

    let n = bestAxis;

    // Reference face on A: most aligned with n
    var refFaceIdx = 0u;
    var refDmax = -1e30;
    for (var fi = 0u; fi < MAX_HULL_FACES; fi++) {
        if (fi >= metaA.faceCount) { break; }
        let fn0 = quatRotate(qA, hullFacePlane(metaA, fi, sA).xyz);
        let dd = dot(fn0, n);
        if (dd > refDmax) { refDmax = dd; refFaceIdx = fi; }
    }

    let refPlane = hullFacePlane(metaA, refFaceIdx, sA);
    let refNormal = quatRotate(qA, refPlane.xyz);
    let refIdxBase = hullFaceIdxBase(metaA, refFaceIdx);
    let refIdxCount = hullFaceIdxCount(metaA, refFaceIdx);

    // Incident face on B: most anti-aligned with n
    var incFaceIdx = 0u;
    var incDmin = 1e30;
    for (var fi = 0u; fi < MAX_HULL_FACES; fi++) {
        if (fi >= metaB.faceCount) { break; }
        let fn0 = quatRotate(qB, hullFacePlane(metaB, fi, sB).xyz);
        let dd = dot(fn0, n);
        if (dd < incDmin) { incDmin = dd; incFaceIdx = fi; }
    }

    let incIdxBase = hullFaceIdxBase(metaB, incFaceIdx);
    let incIdxCount = hullFaceIdxCount(metaB, incFaceIdx);

    // Load incident face vertices
    var clipIn: array<vec3f, MAX_CLIP_VERTS>;
    var clipOut: array<vec3f, MAX_CLIP_VERTS>;
    var clipCount = min(incIdxCount, MAX_CLIP_VERTS);
    for (var v = 0u; v < MAX_CLIP_VERTS; v++) {
        if (v >= clipCount) { break; }
        let vi = hullFaceVertIdx(incIdxBase, v);
        clipIn[v] = posB + quatRotate(qB, hullVertex(metaB, vi) * sB);
    }

    // Clip against reference face edge planes
    for (var ei = 0u; ei < MAX_FACE_VERTS; ei++) {
        if (ei >= refIdxCount) { break; }
        let vi = hullFaceVertIdx(refIdxBase, ei);
        let vj = hullFaceVertIdx(refIdxBase, (ei + 1u) % refIdxCount);
        let va = posA + quatRotate(qA, hullVertex(metaA, vi) * sA);
        let vb = posA + quatRotate(qA, hullVertex(metaA, vj) * sA);
        let edge0 = va - vb;
        let planeN = -cross(edge0, refNormal);
        let planeD = dot(va, planeN);
        var outCount = 0u;
        var a = clipIn[clipCount - 1u];
        var da = dot(planeN, a) - planeD;
        for (var v = 0u; v < MAX_CLIP_VERTS; v++) {
            if (v >= clipCount) { break; }
            let b = clipIn[v];
            let db = dot(planeN, b) - planeD;
            if ((da <= 1e-5) != (db <= 1e-5)) {
                var t = 0.0;
                let denom = da - db;
                if (abs(denom) > 1e-6) { t = clamp(da / denom, 0.0, 1.0); }
                if (outCount < MAX_CLIP_VERTS) { clipOut[outCount] = a + (b - a) * t; outCount++; }
            }
            if (db <= 1e-5) {
                if (outCount < MAX_CLIP_VERTS) { clipOut[outCount] = b; outCount++; }
            }
            a = b;
            da = db;
        }
        clipCount = outCount;
        for (var v = 0u; v < MAX_CLIP_VERTS; v++) {
            if (v >= clipCount) { break; }
            clipIn[v] = clipOut[v];
        }
    }

    // Collect all candidates behind reference face
    let localPlaneEq = refPlane.w;
    let worldPlaneEq = localPlaneEq - dot(refNormal, posA);
    var candidates: array<ManifoldCandidate, MAX_CANDIDATES>;
    var candCount = 0u;

    for (var v = 0u; v < MAX_CLIP_VERTS; v++) {
        if (v >= clipCount) { break; }
        let depth = dot(refNormal, clipIn[v]) + worldPlaneEq;
        if (depth <= 0.0 && candCount < MAX_CANDIDATES) {
            let pB0 = clipIn[v];
            let pA0 = pB0 - refNormal * depth;
            candidates[candCount] = ManifoldCandidate(pA0, pB0, depth, v);
            candCount++;
        }
    }

    // Reduce to 4 well-distributed contacts
    var selected: array<u32, 4>;
    let satCount = reduceManifold(&candidates, candCount, n, &selected);

    let bbBasisN = -n;
    let tb = tangentBasis(bbBasisN);
    let mu = sqrt(bA.friction * bB.friction);

    for (var s = 0u; s < satCount; s++) {
        let c = candidates[selected[s]];
        let rA_w = c.pointA - posA;
        let rB_w = c.pointB - posB;
        let rA = quatRotate(quatConj(qA), rA_w);
        let rB = quatRotate(quatConj(qB), rB_w);
        let bbCI = contactCInit(posA, rA_w, posB, rB_w, bbBasisN, tb[0], tb[1]);

        let fkey = (11u << 24u) | s;
        let wsKey = packKey(ai, bi, s);
        pushConstraintSearching(
            ai, i32(bi), fkey,
            bbBasisN, bbCI.x,
            tb[0], bbCI.y,
            tb[1], bbCI.z,
            rA, rB,
            mu,
            wsKey, ai, bi, 0u,
            -1e30, 0.0, 1e30,
        );
    }
    for (var s = satCount; s < MAX_PAIR_CONTACTS; s++) {
        resetWarmstartHash(packKey(ai, bi, s));
    }
}
`;

const PAIR_TYPE_NAMES = [
    "box-box",
    "sphere-box",
    "capsule-box",
    "sphere-sphere",
    "capsule-sphere",
    "capsule-capsule",
    "hull-box",
    "hull-sphere",
    "hull-capsule",
    "hull-hull",
];

const PAIR_TYPE_DETECTION = [
    detectBoxBoxWGSL,
    detectSphereBoxWGSL,
    detectCapsuleBoxWGSL,
    detectSphereSphereWGSL,
    detectCapsuleSphereWGSL,
    detectCapsuleCapsuleWGSL,
    detectHullBoxWGSL,
    detectHullSphereWGSL,
    detectHullCapsuleWGSL,
    detectHullHullWGSL,
];

const PAIR_TYPE_ENTRY_FN = [
    "detectBoxBox",
    "detectSphereBox",
    "detectCapsuleBox",
    "detectSphereSphere",
    "detectCapsuleSphere",
    "detectCapsuleCapsule",
    "detectHullBox",
    "detectHullSphere",
    "detectHullCapsule",
    "detectHullHull",
];

export { PAIR_TYPE_NAMES };

export function narrowphaseWGSL(pairType: number) {
    return /* wgsl */ `
${sharedNarrowphaseWGSL}
${PAIR_TYPE_DETECTION[pairType]}

const PAIR_TYPE: u32 = ${pairType}u;

@compute @workgroup_size(64)
fn narrowphase(@builtin(global_invocation_id) gid: vec3u) {
    let typeCount = atomicLoad(&solverState[${SS_PAIR_TYPE_BASE}u + PAIR_TYPE]);
    if (gid.x >= typeCount) { return; }
    let maxPerType = params.capacity * params.constraintMul;
    let base = PAIR_TYPE * maxPerType;
    let pairA = pairs[(base + gid.x) * 2u];
    let pairB = pairs[(base + gid.x) * 2u + 1u];
    ${PAIR_TYPE_ENTRY_FN[pairType]}(pairA, pairB);
}
`;
}
