import { BODY_STRIDE, BODY_QUAT_OFFSET } from "./utility.wgsl";

export const interpolateWGSL = /* wgsl */ `
struct InterpParams {
    alpha: f32,
    bodyCount: u32,
}

@group(0) @binding(0) var<storage, read> prevBodies: array<f32>;
@group(0) @binding(1) var<storage, read> currentBodies: array<f32>;
@group(0) @binding(2) var<storage, read> bodyEids: array<u32>;
@group(0) @binding(3) var<uniform> params: InterpParams;
@group(0) @binding(4) var<storage, read_write> matrices: array<f32>;

const BODY_STRIDE: u32 = ${BODY_STRIDE}u;
const QUAT_OFFSET: u32 = ${BODY_QUAT_OFFSET}u;

fn quatToMat(qx: f32, qy: f32, qz: f32, qw: f32) -> array<f32, 9> {
    let x2 = qx + qx;
    let y2 = qy + qy;
    let z2 = qz + qz;
    let xx = qx * x2;
    let xy = qx * y2;
    let xz = qx * z2;
    let yy = qy * y2;
    let yz = qy * z2;
    let zz = qz * z2;
    let wx = qw * x2;
    let wy = qw * y2;
    let wz = qw * z2;
    return array<f32, 9>(
        1.0 - yy - zz, xy + wz, xz - wy,
        xy - wz, 1.0 - xx - zz, yz + wx,
        xz + wy, yz - wx, 1.0 - xx - yy,
    );
}

@compute @workgroup_size(64)
fn interpolate(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.bodyCount) { return; }

    let alpha = params.alpha;
    let off = i * BODY_STRIDE;

    let px = mix(prevBodies[off], currentBodies[off], alpha);
    let py = mix(prevBodies[off + 1u], currentBodies[off + 1u], alpha);
    let pz = mix(prevBodies[off + 2u], currentBodies[off + 2u], alpha);

    let qOff = off + QUAT_OFFSET;
    let pqx = prevBodies[qOff];
    let pqy = prevBodies[qOff + 1u];
    let pqz = prevBodies[qOff + 2u];
    let pqw = prevBodies[qOff + 3u];
    let cqx = currentBodies[qOff];
    let cqy = currentBodies[qOff + 1u];
    let cqz = currentBodies[qOff + 2u];
    let cqw = currentBodies[qOff + 3u];
    let qdot = pqx * cqx + pqy * cqy + pqz * cqz + pqw * cqw;
    let flip = select(1.0, -1.0, qdot < 0.0);
    var qx = mix(pqx * flip, cqx, alpha);
    var qy = mix(pqy * flip, cqy, alpha);
    var qz = mix(pqz * flip, cqz, alpha);
    var qw = mix(pqw * flip, cqw, alpha);

    let qLen = sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (qLen > 1e-12) {
        let invLen = 1.0 / qLen;
        qx *= invLen;
        qy *= invLen;
        qz *= invLen;
        qw *= invLen;
    } else {
        qx = 0.0;
        qy = 0.0;
        qz = 0.0;
        qw = 1.0;
    }

    let rot = quatToMat(qx, qy, qz, qw);

    let eid = bodyEids[i];
    let mOff = eid * 16u;
    matrices[mOff]      = rot[0];
    matrices[mOff + 1u] = rot[1];
    matrices[mOff + 2u] = rot[2];
    matrices[mOff + 3u] = 0.0;
    matrices[mOff + 4u] = rot[3];
    matrices[mOff + 5u] = rot[4];
    matrices[mOff + 6u] = rot[5];
    matrices[mOff + 7u] = 0.0;
    matrices[mOff + 8u] = rot[6];
    matrices[mOff + 9u] = rot[7];
    matrices[mOff + 10u] = rot[8];
    matrices[mOff + 11u] = 0.0;
    matrices[mOff + 12u] = px;
    matrices[mOff + 13u] = py;
    matrices[mOff + 14u] = pz;
    matrices[mOff + 15u] = 1.0;
}
`;
