import { clamp, compose, lerp, multiply, slerp } from "../../engine";

// Vertex Animation Texture bake — the deviceless, pure half of the skinned/animated glTF path.
// A skinned clip is baked to per-frame
// object-space positions + normals by CPU linear-blend skinning, so the skeleton dissolves at bake
// time like a static node chain (the importer already bakes static node hierarchies to flat
// transforms). The GPU half (index.ts) encodes the frames into filterable textures the skin surface
// samples in its `vs`. Input is glTF-agnostic ({@link SkinInput}) so the math is testable against
// hand-computed LBS without a glTF fixture; gltf.ts decodes a document into it. Decode authority for
// the skinning math: three.js GLTFLoader; decode/encoding reference for the GPU
// side: keijiro HdrpVatExample VATHelper.hlsl.

/** one animation channel: a node's TRS path sampled over keyframes. `values` is 3 floats/key for
 *  translation+scale, 4 (xyzw quaternion) for rotation. STEP holds each key to the next; else LINEAR. */
export interface SkinChannel {
    node: number;
    path: "translation" | "rotation" | "scale";
    times: Float32Array;
    values: Float32Array;
    step: boolean;
}

/** the glTF-agnostic bake input: the node hierarchy's base local TRS (channels override per frame),
 *  the scene roots to walk from, the clip's channels, the skin's joints + inverse-bind matrices, and
 *  the primitive's per-vertex skin weights + rest geometry. Pure data: {@link bakeVat} needs nothing
 *  else, and a test builds a 2-bone rig by hand. */
export interface SkinInput {
    nodes: {
        t: [number, number, number];
        r: [number, number, number, number];
        s: [number, number, number];
        children: number[];
    }[];
    roots: number[];
    channels: SkinChannel[];
    /** node indices, in skin-joint order: `inverseBind[j]` + the deform's joint slots index this */
    joints: number[];
    /** column-major mat4 per joint, `joints.length * 16` */
    inverseBind: Float32Array;
    /** per-vertex 4 joint slots (indices into `joints`), `vertCount * 4` */
    jointIndex: Uint16Array;
    /** per-vertex 4 weights, `vertCount * 4` */
    weights: Float32Array;
    /** rest-pose positions, `vertCount * 3`, object space */
    restPos: Float32Array;
    /** rest-pose normals, `vertCount * 3`, unit */
    restNormal: Float32Array;
    /** clip length in seconds (max channel keyframe time) */
    duration: number;
}

/** one baked clip: per-frame object-space positions + normals (frame-major, `frameCount × vertCount`)
 *  and the conservative all-frames AABB the importer turns into the mesh's cull bound. `fps` is the
 *  effective sample rate `(frameCount-1)/duration`: the skin surface multiplies play-time by it to
 *  land on a fractional frame, hardware-lerped across the two adjacent rows. */
export interface GltfVat {
    frameCount: number;
    fps: number;
    duration: number;
    vertCount: number;
    positions: Float32Array;
    normals: Float32Array;
    aabb: { min: [number, number, number]; max: [number, number, number] };
}

// sample one channel's value at time t into `out` (3 or 4 comps). Clamps to the key range; rotation
// LINEAR is slerp (the others component lerp); STEP holds the lower key.
function sampleChannel(ch: SkinChannel, t: number, out: Float32Array): void {
    const comps = ch.path === "rotation" ? 4 : 3;
    const n = ch.times.length;
    if (n === 0) return;
    if (n === 1 || t <= ch.times[0]) {
        for (let c = 0; c < comps; c++) out[c] = ch.values[c];
        return;
    }
    if (t >= ch.times[n - 1]) {
        const o = (n - 1) * comps;
        for (let c = 0; c < comps; c++) out[c] = ch.values[o + c];
        return;
    }
    // binary search for the segment [i, i+1] with times[i] <= t < times[i+1]
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (ch.times[mid] <= t) lo = mid;
        else hi = mid;
    }
    const a = lo * comps;
    const b = (lo + 1) * comps;
    if (ch.step) {
        for (let c = 0; c < comps; c++) out[c] = ch.values[a + c];
        return;
    }
    const u = (t - ch.times[lo]) / (ch.times[lo + 1] - ch.times[lo]);
    if (comps === 4) {
        const q = slerp(
            ch.values[a],
            ch.values[a + 1],
            ch.values[a + 2],
            ch.values[a + 3],
            ch.values[b],
            ch.values[b + 1],
            ch.values[b + 2],
            ch.values[b + 3],
            u,
        );
        out[0] = q.x;
        out[1] = q.y;
        out[2] = q.z;
        out[3] = q.w;
    } else {
        for (let c = 0; c < 3; c++) out[c] = lerp(ch.values[a + c], ch.values[b + c], u);
    }
}

/**
 * bake a skinned clip to flat per-frame vertex data (VAT). Linear-blend skins every vertex at each
 * sampled frame, producing object-space positions + normals + the conservative all-frames AABB. Pure:
 * the heavy frame×vertex loop, kept off the deviceless conformance walk and run only on the GPU load
 * path. `fps` is the target sample rate (subsampled if the clip would exceed `maxFrames`).
 *
 * @example
 * const vat = bakeVat(skinInput, { fps: 30 });
 */
export function bakeVat(
    input: SkinInput,
    opts: { fps?: number; maxFrames?: number } = {},
): GltfVat {
    const fps = opts.fps ?? 30;
    const maxFrames = opts.maxFrames ?? 120;
    const vertCount = input.restPos.length / 3;
    const duration = input.duration;
    const rawFrames = Math.round(duration * fps) + 1;
    const frameCount = duration > 0 ? clamp(rawFrames, 2, maxFrames) : 1;
    const sampleFps = frameCount > 1 ? (frameCount - 1) / duration : 0;
    // a clip past the cap subsamples to fit — warn so a decimated animation isn't shipped silently (raise
    // maxFrames for the full rate). Decodes are cached, so this fires once per asset.
    if (duration > 0 && rawFrames > maxFrames) {
        console.warn(
            `[gltf] VAT clip ${duration.toFixed(2)}s exceeds the ${maxFrames}-frame cap at ${fps}fps — ` +
                `subsampled to ${sampleFps.toFixed(1)}fps (raise maxFrames for the full rate)`,
        );
    }

    const nodeCount = input.nodes.length;
    const jointCount = input.joints.length;
    const local = new Float32Array(nodeCount * 16);
    const global = new Float32Array(nodeCount * 16);
    const skin = new Float32Array(jointCount * 16);
    const lm = new Float32Array(16);
    const trs = new Float32Array(4); // channel sample scratch (max 4 comps)

    const positions = new Float32Array(frameCount * vertCount * 3);
    const normals = new Float32Array(frameCount * vertCount * 3);
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];

    // channels grouped by node, so a frame overrides only the animated nodes' TRS off the base
    const byNode: SkinChannel[][] = Array.from({ length: nodeCount }, () => []);
    for (const ch of input.channels) byNode[ch.node]?.push(ch);

    // hierarchy-walk scratch, frame-invariant
    const identity = compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, new Float32Array(16));
    const parentMat = new Float32Array(16);
    const gm = new Float32Array(16);
    const stack: { node: number; parent: number }[] = [];

    for (let f = 0; f < frameCount; f++) {
        const t = frameCount > 1 ? (f / (frameCount - 1)) * duration : 0;

        // 1. local matrices: base TRS, animated paths overridden
        for (let n = 0; n < nodeCount; n++) {
            const node = input.nodes[n];
            const tx = [node.t[0], node.t[1], node.t[2]];
            const rot = [node.r[0], node.r[1], node.r[2], node.r[3]];
            const sc = [node.s[0], node.s[1], node.s[2]];
            for (const ch of byNode[n]) {
                sampleChannel(ch, t, trs);
                if (ch.path === "translation") {
                    tx[0] = trs[0];
                    tx[1] = trs[1];
                    tx[2] = trs[2];
                } else if (ch.path === "rotation") {
                    rot[0] = trs[0];
                    rot[1] = trs[1];
                    rot[2] = trs[2];
                    rot[3] = trs[3];
                } else {
                    sc[0] = trs[0];
                    sc[1] = trs[1];
                    sc[2] = trs[2];
                }
            }
            compose(tx[0], tx[1], tx[2], rot[0], rot[1], rot[2], rot[3], sc[0], sc[1], sc[2], lm);
            local.set(lm, n * 16);
        }

        // 2. global matrices: walk the hierarchy from the roots (stack, parent-applied)
        stack.length = 0;
        for (let i = input.roots.length - 1; i >= 0; i--) {
            stack.push({ node: input.roots[i], parent: -1 });
        }
        while (stack.length > 0) {
            const { node, parent } = stack.pop()!;
            if (parent < 0) parentMat.set(identity);
            else parentMat.set(global.subarray(parent * 16, parent * 16 + 16));
            multiply(parentMat, local.subarray(node * 16, node * 16 + 16), gm);
            global.set(gm, node * 16);
            for (const child of input.nodes[node].children)
                stack.push({ node: child, parent: node });
        }

        // 3. skin matrices: jointGlobal · inverseBind (glTF 3.7.3; mesh-node transform ignored)
        for (let j = 0; j < jointCount; j++) {
            multiply(
                global.subarray(input.joints[j] * 16, input.joints[j] * 16 + 16),
                input.inverseBind.subarray(j * 16, j * 16 + 16),
                skin.subarray(j * 16, j * 16 + 16),
            );
        }

        // 4. linear-blend skin every vertex: p' = Σ wᵢ·(Mᵢ·p), n' = normalize(Σ wᵢ·(Mᵢ₃ₓ₃·n))
        const base = f * vertCount * 3;
        for (let v = 0; v < vertCount; v++) {
            const px = input.restPos[v * 3];
            const py = input.restPos[v * 3 + 1];
            const pz = input.restPos[v * 3 + 2];
            const nx = input.restNormal[v * 3];
            const ny = input.restNormal[v * 3 + 1];
            const nz = input.restNormal[v * 3 + 2];
            let ox = 0;
            let oy = 0;
            let oz = 0;
            let onx = 0;
            let ony = 0;
            let onz = 0;
            let wsum =
                input.weights[v * 4] +
                input.weights[v * 4 + 1] +
                input.weights[v * 4 + 2] +
                input.weights[v * 4 + 3];
            if (wsum <= 0) wsum = 1; // an unweighted vertex rides joint 0 at weight 0 → stays at rest
            for (let k = 0; k < 4; k++) {
                const w = input.weights[v * 4 + k] / wsum;
                if (w === 0) continue;
                const m = input.jointIndex[v * 4 + k] * 16;
                ox += w * (skin[m] * px + skin[m + 4] * py + skin[m + 8] * pz + skin[m + 12]);
                oy += w * (skin[m + 1] * px + skin[m + 5] * py + skin[m + 9] * pz + skin[m + 13]);
                oz += w * (skin[m + 2] * px + skin[m + 6] * py + skin[m + 10] * pz + skin[m + 14]);
                onx += w * (skin[m] * nx + skin[m + 4] * ny + skin[m + 8] * nz);
                ony += w * (skin[m + 1] * nx + skin[m + 5] * ny + skin[m + 9] * nz);
                onz += w * (skin[m + 2] * nx + skin[m + 6] * ny + skin[m + 10] * nz);
            }
            const o = base + v * 3;
            positions[o] = ox;
            positions[o + 1] = oy;
            positions[o + 2] = oz;
            const nl = Math.hypot(onx, ony, onz) || 1;
            normals[o] = onx / nl;
            normals[o + 1] = ony / nl;
            normals[o + 2] = onz / nl;
            if (ox < min[0]) min[0] = ox;
            if (oy < min[1]) min[1] = oy;
            if (oz < min[2]) min[2] = oz;
            if (ox > max[0]) max[0] = ox;
            if (oy > max[1]) max[1] = oy;
            if (oz > max[2]) max[2] = oz;
        }
    }

    if (!Number.isFinite(min[0])) {
        min[0] = min[1] = min[2] = 0;
        max[0] = max[1] = max[2] = 0;
    }
    return {
        frameCount,
        fps: sampleFps,
        duration,
        vertCount,
        positions,
        normals,
        aabb: { min, max },
    };
}
