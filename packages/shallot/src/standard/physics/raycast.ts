import { Shape, type Vec3, type Ray, type State } from "../../engine";
import { Transform, WorldTransform } from "../transforms";
import { Camera, Part, Mesh, getMesh, type MeshData } from "../render";

export type { Vec3, Ray };

export interface Hit {
    eid: number;
    distance: number;
    point: Vec3;
    normal: Vec3;
}

const DEG_TO_RAD = Math.PI / 180;

export function generateRay(
    screenX: number,
    screenY: number,
    width: number,
    height: number,
    fov: number,
    near: number,
    cameraWorld: Float32Array,
): Ray {
    const ndcX = screenX * 2 - 1;
    const ndcY = 1 - screenY * 2;

    const aspect = width / height;
    const tanHalfFov = Math.tan((fov * DEG_TO_RAD) / 2);

    const camDirX = ndcX * aspect * tanHalfFov;
    const camDirY = ndcY * tanHalfFov;
    const camDirZ = -1;

    const r00 = cameraWorld[0];
    const r10 = cameraWorld[1];
    const r20 = cameraWorld[2];
    const r01 = cameraWorld[4];
    const r11 = cameraWorld[5];
    const r21 = cameraWorld[6];
    const r02 = cameraWorld[8];
    const r12 = cameraWorld[9];
    const r22 = cameraWorld[10];

    let dirX = r00 * camDirX + r01 * camDirY + r02 * camDirZ;
    let dirY = r10 * camDirX + r11 * camDirY + r12 * camDirZ;
    let dirZ = r20 * camDirX + r21 * camDirY + r22 * camDirZ;

    const len = Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
    dirX /= len;
    dirY /= len;
    dirZ /= len;

    const originX = cameraWorld[12];
    const originY = cameraWorld[13];
    const originZ = cameraWorld[14];

    return {
        origin: {
            x: originX + dirX * near,
            y: originY + dirY * near,
            z: originZ + dirZ * near,
        },
        direction: { x: dirX, y: dirY, z: dirZ },
    };
}

export function screenToRay(
    state: State,
    screenX: number,
    screenY: number,
    width: number,
    height: number,
): Ray | null {
    for (const eid of state.query([Camera])) {
        if (!Camera.active[eid]) continue;
        const cameraWorld = WorldTransform.data.subarray(eid * 16, eid * 16 + 16);
        return generateRay(
            screenX,
            screenY,
            width,
            height,
            Camera.fov[eid],
            Camera.near[eid],
            cameraWorld,
        );
    }
    return null;
}

export function raySphere(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
): { t: number; nx: number; ny: number; nz: number } | null {
    const lx = ox - cx;
    const ly = oy - cy;
    const lz = oz - cz;

    const a = dx * dx + dy * dy + dz * dz;
    const b = 2 * (lx * dx + ly * dy + lz * dz);
    const c = lx * lx + ly * ly + lz * lz - radius * radius;

    const disc = b * b - 4 * a * c;
    if (disc < 0) return null;

    const sqrtDisc = Math.sqrt(disc);
    let t = (-b - sqrtDisc) / (2 * a);
    if (t < 0) {
        t = (-b + sqrtDisc) / (2 * a);
        if (t < 0) return null;
    }

    const px = ox + dx * t - cx;
    const py = oy + dy * t - cy;
    const pz = oz + dz * t - cz;
    const len = Math.sqrt(px * px + py * py + pz * pz);

    return { t, nx: px / len, ny: py / len, nz: pz / len };
}

function quatRotate(
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    vx: number,
    vy: number,
    vz: number,
): [number, number, number] {
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    return [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ];
}

export function rayCapsule(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
    halfHeight: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
): { t: number; nx: number; ny: number; nz: number } | null {
    const lox = ox - cx;
    const loy = oy - cy;
    const loz = oz - cz;

    const [rox, roy, roz] = quatRotate(-qx, -qy, -qz, qw, lox, loy, loz);
    const [rdx, rdy, rdz] = quatRotate(-qx, -qy, -qz, qw, dx, dy, dz);

    let bestT = Infinity;
    let bestNx = 0;
    let bestNy = 0;
    let bestNz = 0;

    // cylinder body (infinite cylinder along Y, clamped to [-halfHeight, halfHeight])
    const a = rdx * rdx + rdz * rdz;
    const b = 2 * (rox * rdx + roz * rdz);
    const c = rox * rox + roz * roz - radius * radius;
    const disc = b * b - 4 * a * c;

    if (a > 1e-12 && disc >= 0) {
        const sqrtDisc = Math.sqrt(disc);
        for (const sign of [-1, 1]) {
            const t = (-b + sign * sqrtDisc) / (2 * a);
            if (t >= 0 && t < bestT) {
                const hitY = roy + rdy * t;
                if (hitY >= -halfHeight && hitY <= halfHeight) {
                    const hitX = rox + rdx * t;
                    const hitZ = roz + rdz * t;
                    const len = Math.sqrt(hitX * hitX + hitZ * hitZ);
                    if (len > 1e-12) {
                        bestT = t;
                        bestNx = hitX / len;
                        bestNy = 0;
                        bestNz = hitZ / len;
                    }
                }
            }
        }
    }

    // hemisphere caps at y = +halfHeight and y = -halfHeight
    for (const capSign of [-1, 1]) {
        const capY = capSign * halfHeight;
        const slx = rox;
        const sly = roy - capY;
        const slz = roz;
        const sa = rdx * rdx + rdy * rdy + rdz * rdz;
        const sb = 2 * (slx * rdx + sly * rdy + slz * rdz);
        const sc = slx * slx + sly * sly + slz * slz - radius * radius;
        const sd = sb * sb - 4 * sa * sc;
        if (sd >= 0) {
            const sqrtSd = Math.sqrt(sd);
            for (const sign of [-1, 1]) {
                const t = (-sb + sign * sqrtSd) / (2 * sa);
                if (t >= 0 && t < bestT) {
                    const hitY = roy + rdy * t - capY;
                    if (capSign * hitY >= 0) {
                        const hnx = rox + rdx * t;
                        const hny = hitY;
                        const hnz = roz + rdz * t;
                        const len = Math.sqrt(hnx * hnx + hny * hny + hnz * hnz);
                        if (len > 1e-12) {
                            bestT = t;
                            bestNx = hnx / len;
                            bestNy = hny / len;
                            bestNz = hnz / len;
                        }
                    }
                }
            }
        }
    }

    if (bestT === Infinity) return null;

    const [nx, ny, nz] = quatRotate(qx, qy, qz, qw, bestNx, bestNy, bestNz);
    return { t: bestT, nx, ny, nz };
}

export function rayOBB(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    cx: number,
    cy: number,
    cz: number,
    hx: number,
    hy: number,
    hz: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
): { t: number; nx: number; ny: number; nz: number } | null {
    const lox = ox - cx;
    const loy = oy - cy;
    const loz = oz - cz;

    const [rox, roy, roz] = quatRotate(-qx, -qy, -qz, qw, lox, loy, loz);
    const [rdx, rdy, rdz] = quatRotate(-qx, -qy, -qz, qw, dx, dy, dz);

    let tmin = -Infinity;
    let tmax = Infinity;
    let normalAxis = 0;
    let normalSign = 1;

    for (let i = 0; i < 3; i++) {
        const o = i === 0 ? rox : i === 1 ? roy : roz;
        const d = i === 0 ? rdx : i === 1 ? rdy : rdz;
        const h = i === 0 ? hx : i === 1 ? hy : hz;
        if (Math.abs(d) < 1e-12) {
            if (o < -h || o > h) return null;
        } else {
            const invD = 1 / d;
            let t1 = (-h - o) * invD;
            let t2 = (h - o) * invD;
            let sign = -1;
            if (t1 > t2) {
                const tmp = t1;
                t1 = t2;
                t2 = tmp;
                sign = 1;
            }
            if (t1 > tmin) {
                tmin = t1;
                normalAxis = i;
                normalSign = sign;
            }
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return null;
        }
    }

    if (tmax < 0) return null;

    const t = tmin >= 0 ? tmin : tmax;
    if (t < 0) return null;

    const lnx = normalAxis === 0 ? normalSign : 0;
    const lny = normalAxis === 1 ? normalSign : 0;
    const lnz = normalAxis === 2 ? normalSign : 0;
    const [nx, ny, nz] = quatRotate(qx, qy, qz, qw, lnx, lny, lnz);

    return { t, nx, ny, nz };
}

export function rayTriangle(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    v0x: number,
    v0y: number,
    v0z: number,
    v1x: number,
    v1y: number,
    v1z: number,
    v2x: number,
    v2y: number,
    v2z: number,
): { t: number; nx: number; ny: number; nz: number } | null {
    const e1x = v1x - v0x,
        e1y = v1y - v0y,
        e1z = v1z - v0z;
    const e2x = v2x - v0x,
        e2y = v2y - v0y,
        e2z = v2z - v0z;

    const hx = dy * e2z - dz * e2y;
    const hy = dz * e2x - dx * e2z;
    const hz = dx * e2y - dy * e2x;

    const a = e1x * hx + e1y * hy + e1z * hz;
    if (a > -1e-10 && a < 1e-10) return null;

    const f = 1 / a;
    const sx = ox - v0x,
        sy = oy - v0y,
        sz = oz - v0z;

    const u = f * (sx * hx + sy * hy + sz * hz);
    if (u < 0 || u > 1) return null;

    const qx = sy * e1z - sz * e1y;
    const qy = sz * e1x - sx * e1z;
    const qz = sx * e1y - sy * e1x;

    const v = f * (dx * qx + dy * qy + dz * qz);
    if (v < 0 || u + v > 1) return null;

    const t = f * (e2x * qx + e2y * qy + e2z * qz);
    if (t < 0) return null;

    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const nlen = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (nlen < 1e-14) return null;
    nx /= nlen;
    ny /= nlen;
    nz /= nlen;

    if (nx * dx + ny * dy + nz * dz > 0) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }

    return { t, nx, ny, nz };
}

export function rayMesh(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    cx: number,
    cy: number,
    cz: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    sx: number,
    sy: number,
    sz: number,
    meshData: MeshData,
): { t: number; nx: number; ny: number; nz: number } | null {
    const lox = ox - cx,
        loy = oy - cy,
        loz = oz - cz;
    const [rox, roy, roz] = quatRotate(-qx, -qy, -qz, qw, lox, loy, loz);
    const [rdx, rdy, rdz] = quatRotate(-qx, -qy, -qz, qw, dx, dy, dz);

    const { vertices, indices, indexCount } = meshData;
    let best: { t: number; nx: number; ny: number; nz: number } | null = null;

    for (let i = 0; i < indexCount; i += 3) {
        const i0 = indices[i] * 8;
        const i1 = indices[i + 1] * 8;
        const i2 = indices[i + 2] * 8;

        const hit = rayTriangle(
            rox,
            roy,
            roz,
            rdx,
            rdy,
            rdz,
            vertices[i0] * sx,
            vertices[i0 + 1] * sy,
            vertices[i0 + 2] * sz,
            vertices[i1] * sx,
            vertices[i1 + 1] * sy,
            vertices[i1 + 2] * sz,
            vertices[i2] * sx,
            vertices[i2 + 1] * sy,
            vertices[i2 + 2] * sz,
        );
        if (hit && (!best || hit.t < best.t)) {
            best = hit;
        }
    }

    if (!best) return null;

    const [nx, ny, nz] = quatRotate(qx, qy, qz, qw, best.nx, best.ny, best.nz);
    return { t: best.t, nx, ny, nz };
}

export function raycast(entities: Iterable<number>, ray: Ray): Hit | null {
    const ox = ray.origin.x;
    const oy = ray.origin.y;
    const oz = ray.origin.z;
    const dx = ray.direction.x;
    const dy = ray.direction.y;
    const dz = ray.direction.z;

    let closest: Hit | null = null;

    for (const eid of entities) {
        const shape = Part.shape[eid];
        const px = Transform.posX[eid];
        const py = Transform.posY[eid];
        const pz = Transform.posZ[eid];
        const sx = Part.sizeX[eid];
        const sy = Part.sizeY[eid];
        const sz = Part.sizeZ[eid];

        let result: { t: number; nx: number; ny: number; nz: number } | null = null;

        if (shape === Shape.Sphere) {
            const radius = sx / 2;
            result = raySphere(ox, oy, oz, dx, dy, dz, px, py, pz, radius);
        } else if (shape === Shape.Box) {
            const qx = Transform.quatX[eid];
            const qy = Transform.quatY[eid];
            const qz = Transform.quatZ[eid];
            const qw = Transform.quatW[eid];
            result = rayOBB(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                px,
                py,
                pz,
                sx / 2,
                sy / 2,
                sz / 2,
                qx,
                qy,
                qz,
                qw,
            );
        } else if (shape === Shape.Capsule) {
            const qx = Transform.quatX[eid];
            const qy = Transform.quatY[eid];
            const qz = Transform.quatZ[eid];
            const qw = Transform.quatW[eid];
            result = rayCapsule(ox, oy, oz, dx, dy, dz, px, py, pz, sx / 2, sy / 2, qx, qy, qz, qw);
        } else if (shape === Shape.Mesh) {
            const meshId = Mesh.geometry[eid];
            const meshData = getMesh(meshId);
            if (meshData) {
                const qx = Transform.quatX[eid];
                const qy = Transform.quatY[eid];
                const qz = Transform.quatZ[eid];
                const qw = Transform.quatW[eid];
                result = rayMesh(
                    ox,
                    oy,
                    oz,
                    dx,
                    dy,
                    dz,
                    px,
                    py,
                    pz,
                    qx,
                    qy,
                    qz,
                    qw,
                    sx,
                    sy,
                    sz,
                    meshData,
                );
            }
        }

        if (result && (!closest || result.t < closest.distance)) {
            closest = {
                eid,
                distance: result.t,
                point: {
                    x: ox + dx * result.t,
                    y: oy + dy * result.t,
                    z: oz + dz * result.t,
                },
                normal: { x: result.nx, y: result.ny, z: result.nz },
            };
        }
    }

    return closest;
}
