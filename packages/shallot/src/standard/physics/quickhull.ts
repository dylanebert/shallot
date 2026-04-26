// Quickhull3D — flat-array implementation with half-edge adjacency.
// Reference: mauriciopoppe/quickhull3d (MIT), from John Lloyd's Java quickhull.

const NONE = -1;

// Half-edge fi*3+k: head = hv[fi*3+k], tail = hv[fi*3+(k+2)%3].
// Edge goes tail → head. next = fi*3+(k+1)%3, prev = fi*3+(k+2)%3.
function heNext(he: number): number {
    return he - (he % 3) + (((he % 3) + 1) % 3);
}
function hePrev(he: number): number {
    return he - (he % 3) + (((he % 3) + 2) % 3);
}

interface QH {
    pts: Float64Array;
    hv: Int32Array;
    ho: Int32Array;
    fnx: Float64Array;
    fny: Float64Array;
    fnz: Float64Array;
    foff: Float64Array;
    alive: Uint8Array;
    out: number[][];
    cnt: number;
    cap: number;
    tol: number;
}

function grow(q: QH): void {
    q.cap *= 2;
    const hc = q.cap * 3;
    let t: any;
    t = new Int32Array(hc);
    t.set(q.hv);
    q.hv = t;
    t = new Int32Array(hc);
    t.set(q.ho);
    q.ho = t;
    t = new Float64Array(q.cap);
    t.set(q.fnx);
    q.fnx = t;
    t = new Float64Array(q.cap);
    t.set(q.fny);
    q.fny = t;
    t = new Float64Array(q.cap);
    t.set(q.fnz);
    q.fnz = t;
    t = new Float64Array(q.cap);
    t.set(q.foff);
    q.foff = t;
    t = new Uint8Array(q.cap);
    t.set(q.alive);
    q.alive = t;
}

function computePlane(q: QH, fi: number): void {
    const a = q.hv[fi * 3],
        b = q.hv[fi * 3 + 1],
        c = q.hv[fi * 3 + 2];
    const p = q.pts;
    const ax = p[a * 3],
        ay = p[a * 3 + 1],
        az = p[a * 3 + 2];
    const ux = p[b * 3] - ax,
        uy = p[b * 3 + 1] - ay,
        uz = p[b * 3 + 2] - az;
    const vx = p[c * 3] - ax,
        vy = p[c * 3 + 1] - ay,
        vz = p[c * 3 + 2] - az;
    let nx = uy * vz - uz * vy,
        ny = uz * vx - ux * vz,
        nz = ux * vy - uy * vx;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-15) {
        nx /= len;
        ny /= len;
        nz /= len;
    }
    q.fnx[fi] = nx;
    q.fny[fi] = ny;
    q.fnz[fi] = nz;
    const cx = (p[a * 3] + p[b * 3] + p[c * 3]) / 3;
    const cy = (p[a * 3 + 1] + p[b * 3 + 1] + p[c * 3 + 1]) / 3;
    const cz = (p[a * 3 + 2] + p[b * 3 + 2] + p[c * 3 + 2]) / 3;
    q.foff[fi] = nx * cx + ny * cy + nz * cz;
}

function dist(q: QH, fi: number, pi: number): number {
    return (
        q.fnx[fi] * q.pts[pi * 3] +
        q.fny[fi] * q.pts[pi * 3 + 1] +
        q.fnz[fi] * q.pts[pi * 3 + 2] -
        q.foff[fi]
    );
}

function addFace(q: QH, va: number, vb: number, vc: number): number {
    if (q.cnt >= q.cap) grow(q);
    const fi = q.cnt++;
    const he0 = fi * 3;
    q.hv[he0] = va;
    q.hv[he0 + 1] = vb;
    q.hv[he0 + 2] = vc;
    q.ho[he0] = NONE;
    q.ho[he0 + 1] = NONE;
    q.ho[he0 + 2] = NONE;
    q.alive[fi] = 1;
    while (q.out.length <= fi) q.out.push([]);
    q.out[fi] = [];
    computePlane(q, fi);
    return fi;
}

function link(q: QH, a: number, b: number): void {
    q.ho[a] = b;
    q.ho[b] = a;
}

// Recursive horizon computation matching the reference exactly.
// Walks edges of each visible face in order, producing horizon edges in CCW order.
// crossEdge: the half-edge used to enter this face (on this face's side).
// For the starting face, crossEdge = NONE.
function computeHorizon(
    q: QH,
    eyePt: number,
    crossEdge: number,
    faceIdx: number,
    horizon: number[],
): void {
    // Mark face as deleted, collect its outside points as unclaimed
    q.alive[faceIdx] = 0;

    let edge: number;
    const he0 = faceIdx * 3;
    if (crossEdge === NONE) {
        edge = he0;
    } else {
        // Start from the next edge after crossEdge
        edge = heNext(crossEdge);
    }

    const startEdge = edge;
    do {
        const opp = q.ho[edge];
        if (opp !== NONE) {
            const oppFace = (opp / 3) | 0;
            if (q.alive[oppFace]) {
                if (dist(q, oppFace, eyePt) > q.tol) {
                    computeHorizon(q, eyePt, opp, oppFace, horizon);
                } else {
                    horizon.push(edge);
                }
            }
        }
        edge = heNext(edge);
    } while (edge !== startEdge);
}

export function quickhull(pts: Float64Array, n: number): number[][] {
    if (n < 4) return n === 3 ? [[0, 1, 2]] : [];

    // Extremes
    let mnX = 0,
        mxX = 0,
        mnY = 0,
        mxY = 0,
        mnZ = 0,
        mxZ = 0;
    for (let i = 1; i < n; i++) {
        if (pts[i * 3] < pts[mnX * 3]) mnX = i;
        if (pts[i * 3] > pts[mxX * 3]) mxX = i;
        if (pts[i * 3 + 1] < pts[mnY * 3 + 1]) mnY = i;
        if (pts[i * 3 + 1] > pts[mxY * 3 + 1]) mxY = i;
        if (pts[i * 3 + 2] < pts[mnZ * 3 + 2]) mnZ = i;
        if (pts[i * 3 + 2] > pts[mxZ * 3 + 2]) mxZ = i;
    }

    const tol =
        3 *
        Number.EPSILON *
        (Math.max(Math.abs(pts[mnX * 3]), Math.abs(pts[mxX * 3])) +
            Math.max(Math.abs(pts[mnY * 3 + 1]), Math.abs(pts[mxY * 3 + 1])) +
            Math.max(Math.abs(pts[mnZ * 3 + 2]), Math.abs(pts[mxZ * 3 + 2])));

    // Most distant pair among extremes
    const ext = [mnX, mxX, mnY, mxY, mnZ, mxZ];
    let iA = 0,
        iB = 1,
        bd2 = 0;
    for (let i = 0; i < ext.length; i++) {
        for (let j = i + 1; j < ext.length; j++) {
            const a = ext[i],
                b = ext[j];
            const dx = pts[b * 3] - pts[a * 3],
                dy = pts[b * 3 + 1] - pts[a * 3 + 1],
                dz = pts[b * 3 + 2] - pts[a * 3 + 2];
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > bd2) {
                bd2 = d2;
                iA = a;
                iB = b;
            }
        }
    }

    const ux = pts[iB * 3] - pts[iA * 3],
        uy = pts[iB * 3 + 1] - pts[iA * 3 + 1],
        uz = pts[iB * 3 + 2] - pts[iA * 3 + 2];

    let iC = -1,
        bld = 0;
    for (let i = 0; i < n; i++) {
        if (i === iA || i === iB) continue;
        const px = pts[i * 3] - pts[iA * 3],
            py = pts[i * 3 + 1] - pts[iA * 3 + 1],
            pz = pts[i * 3 + 2] - pts[iA * 3 + 2];
        const cx = uy * pz - uz * py,
            cy = uz * px - ux * pz,
            cz = ux * py - uy * px;
        const d2 = cx * cx + cy * cy + cz * cz;
        if (d2 > bld) {
            bld = d2;
            iC = i;
        }
    }

    const vx = pts[iC * 3] - pts[iA * 3],
        vy = pts[iC * 3 + 1] - pts[iA * 3 + 1],
        vz = pts[iC * 3 + 2] - pts[iA * 3 + 2];
    let pnx = uy * vz - uz * vy,
        pny = uz * vx - ux * vz,
        pnz = ux * vy - uy * vx;
    const pl = Math.sqrt(pnx * pnx + pny * pny + pnz * pnz);
    pnx /= pl;
    pny /= pl;
    pnz /= pl;
    const d0 = pnx * pts[iA * 3] + pny * pts[iA * 3 + 1] + pnz * pts[iA * 3 + 2];

    let iD = -1,
        bpd = 0;
    for (let i = 0; i < n; i++) {
        if (i === iA || i === iB || i === iC) continue;
        const d = Math.abs(pnx * pts[i * 3] + pny * pts[i * 3 + 1] + pnz * pts[i * 3 + 2] - d0);
        if (d > bpd) {
            bpd = d;
            iD = i;
        }
    }
    if (iC === -1 || iD === -1) return [[0, 1, 2]];

    // Orient tetrahedron — match reference's createInitialSimplex exactly
    const orient = pnx * pts[iD * 3] + pny * pts[iD * 3 + 1] + pnz * pts[iD * 3 + 2] - d0;

    const cap = Math.max(64, n * 4);
    const q: QH = {
        pts,
        hv: new Int32Array(cap * 3),
        ho: new Int32Array(cap * 3).fill(NONE),
        fnx: new Float64Array(cap),
        fny: new Float64Array(cap),
        fnz: new Float64Array(cap),
        foff: new Float64Array(cap),
        alive: new Uint8Array(cap),
        out: [],
        cnt: 0,
        cap,
        tol,
    };

    // Reference tetrahedron winding + connectivity (from createInitialSimplex):
    // orient < 0: normal of (v0,v1,v2) points away from v3
    //   f0 = (v0,v1,v2), f1 = (v3,v1,v0), f2 = (v3,v2,v1), f3 = (v3,v0,v2)
    //   opposites: f[i+1].edge(2) ↔ f[0].edge((i+1)%3), f[i+1].edge(1) ↔ f[(i+1)%3+1].edge(0)
    // orient >= 0: flip
    //   f0 = (v0,v2,v1), f1 = (v3,v0,v1), f2 = (v3,v1,v2), f3 = (v3,v2,v0)
    //   opposites: f[i+1].edge(2) ↔ f[0].edge((3-i)%3), f[i+1].edge(0) ↔ f[(i+1)%3+1].edge(1)

    if (orient < 0) {
        addFace(q, iA, iB, iC);
        addFace(q, iD, iB, iA);
        addFace(q, iD, iC, iB);
        addFace(q, iD, iA, iC);
        for (let i = 0; i < 3; i++) {
            const j = (i + 1) % 3;
            link(q, (i + 1) * 3 + 2, j);
            link(q, (i + 1) * 3 + 1, (j + 1) * 3);
        }
    } else {
        addFace(q, iA, iC, iB);
        addFace(q, iD, iA, iB);
        addFace(q, iD, iB, iC);
        addFace(q, iD, iC, iA);
        for (let i = 0; i < 3; i++) {
            const j = (i + 1) % 3;
            link(q, (i + 1) * 3 + 2, (3 - i) % 3);
            link(q, (i + 1) * 3, (j + 1) * 3 + 1);
        }
    }

    // Assign points to initial faces
    const init = new Set([iA, iB, iC, iD]);
    for (let i = 0; i < n; i++) {
        if (init.has(i)) continue;
        let bestFi = -1,
            bestDist = tol;
        for (let f = 0; f < q.cnt; f++) {
            const d = dist(q, f, i);
            if (d > bestDist) {
                bestDist = d;
                bestFi = f;
            }
        }
        if (bestFi >= 0) q.out[bestFi].push(i);
    }

    let maxIter = n * n;
    while (maxIter-- > 0) {
        let curFace = -1;
        for (let f = 0; f < q.cnt; f++) {
            if (q.alive[f] && q.out[f].length > 0) {
                curFace = f;
                break;
            }
        }
        if (curFace === -1) break;

        let eye = -1,
            eyeDist = 0;
        for (const pi of q.out[curFace]) {
            const d = dist(q, curFace, pi);
            if (d > eyeDist) {
                eyeDist = d;
                eye = pi;
            }
        }
        if (eye === -1) break;

        const beforeCnt = q.cnt;
        const horizon: number[] = [];

        const savedOut: number[][] = [];
        for (let f = 0; f < q.cnt; f++) {
            if (q.alive[f]) savedOut[f] = q.out[f];
        }

        computeHorizon(q, eye, NONE, curFace, horizon);

        const orphans: number[] = [];
        for (let f = 0; f < beforeCnt; f++) {
            if (!q.alive[f] && savedOut[f]) {
                for (const p of savedOut[f]) {
                    if (p !== eye) orphans.push(p);
                }
                q.out[f] = [];
            }
        }

        // Create new faces from horizon edges to eye.
        // Reference: addAdjoiningFace creates (eye, tail, head) per horizon edge.
        // he2 (tail→head) links to horizonEdge.opposite.
        // he0 (head→eye) is the "sideEdge".
        // Adjacent faces: current.he1 (eye→tail) ↔ previous.he0 (head→eye).
        const newFaces: number[] = [];
        let firstHe0 = NONE,
            prevHe0 = NONE;
        for (const he of horizon) {
            const head = q.hv[he];
            const tail = q.hv[hePrev(he)];
            const newFi = addFace(q, eye, tail, head);
            newFaces.push(newFi);

            const opp = q.ho[he];
            if (opp !== NONE) link(q, newFi * 3 + 2, opp);

            const curHe0 = newFi * 3;
            if (firstHe0 === NONE) {
                firstHe0 = curHe0;
            } else {
                link(q, newFi * 3 + 1, prevHe0);
            }
            prevHe0 = curHe0;
        }
        // Close the loop: first.he1 ↔ last.he0
        if (firstHe0 !== NONE && prevHe0 !== NONE) {
            link(q, newFaces[0] * 3 + 1, prevHe0);
        }

        // Reassign orphans to new faces
        for (const pi of orphans) {
            let bestFace = -1,
                bestDist = tol;
            for (const newFi of newFaces) {
                const d = dist(q, newFi, pi);
                if (d > bestDist) {
                    bestDist = d;
                    bestFace = newFi;
                }
            }
            if (bestFace >= 0) q.out[bestFace].push(pi);
        }
    }

    const result: number[][] = [];
    for (let f = 0; f < q.cnt; f++) {
        if (q.alive[f]) result.push([q.hv[f * 3], q.hv[f * 3 + 1], q.hv[f * 3 + 2]]);
    }
    return result;
}
