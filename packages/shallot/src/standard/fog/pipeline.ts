// The fog march's typed pipeline: the two bind-group layouts (group 0 — the per-camera scene/depth/output
// + the View/Fog uniforms; group 1 — the camera-independent light + shadow service) and the compute kernel
// over them. The march primitives themselves (`fogDensity`, `fogTransmittance`, the in-scatter terms) are
// pure TGSL fns in `./march`, spliced by both this pipeline and the CPU oracles — this file is the
// pipeline/layout half `march.ts`'s header describes.
//
// Six of the ten group-1 bindings (`pointAtlas`/`shadowSamp`/`pointShadows`/`tileRects` for the point path,
// `shadowMap`/`sunShadow` for the sun path) are read only inside sear's relocatable shadow receivers
// (`pointShadowOf` / `sampleSunShadow`, `sear/core`), which are WGSL-bodied and read those six as free
// names — a real-reference call into them reaches `tgpu.resolve`'s call graph (the call itself is tracked),
// but the free names inside their bodies are not (the `hullData` forcing-touch precedent, `avbd.md`), so
// nothing here would otherwise force their WGSL declarations. `fogKernel` below forces them into scope with
// a read folded into the real, already-live `t` accumulator — a separately discarded local risks the
// JS→WGSL transpiler pruning it as dead, so the fold must feed a value that's genuinely written to `output`.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { clusterCell, LightingGpu, PointLightGpu, PointLights, View } from "../render/core";
import {
    pointCasters,
    pointCastersSchema,
    pointShadowRef,
    SunShadow,
    sampleSunShadow,
    tileRectsSchema,
} from "../sear/core";
import {
    FOG_MAX_STEPS,
    FogGpu,
    fogComposite,
    fogDensity,
    ign,
    inScatterContribution,
    reconstructWorld,
    sunInScatter,
    WORKGROUP,
} from "./march";

/** group 0: the per-camera resources — the resolved scene + depth to march, the storage-texture write
 *  target, and the `View` / `Fog` uniforms. Rebuilt per camera (`index.ts`'s `_views` cache). @internal */
export const fogLayout0 = tgpu
    .bindGroupLayout({
        // "sceneTex", not "scene" — fogComposite's own `scene` parameter would collide with a binding
        // named identically once both land in the same resolve, forcing a confusing rename on one of them
        sceneTex: { texture: d.texture2d(d.f32), visibility: ["compute"] },
        depthTex: { texture: d.textureDepth2d(), visibility: ["compute"] },
        output: {
            storageTexture: d.textureStorage2d("rgba16float", "write-only"),
            visibility: ["compute"],
        },
        view: { uniform: View, visibility: ["compute"] },
        fog: { uniform: FogGpu, visibility: ["compute"] },
    })
    .$idx(0);

// group 1's caster-cap-sized uniforms are built once here (a schema instance is exactly one WGSL struct
// declaration per resolve, and this pipeline's resolve is independent of sear's own raw one — `chunk()`'s
// duplicate-name suffixing only matters *within* one resolve, so a second `PointCasters`/`TileRects`
// instance here doesn't collide with sear's).
const _pointCastersGpu = pointCastersSchema();
const _tileRectsGpu = tileRectsSchema(pointCasters() * 6);

/** group 1: the camera-independent light + shadow service — render's compacted lights + light grid, sear's
 *  point atlas + caster uniform + comparison sampler + sun shadow map + params, and the `Lighting` UBO.
 *  Cached across cameras on resource identity (`index.ts`'s `_lights`). @internal */
export const fogLayout1 = tgpu
    .bindGroupLayout({
        pointLights: { storage: PointLights, access: "readonly", visibility: ["compute"] },
        lightGrid: { storage: d.arrayOf(d.vec2u), access: "readonly", visibility: ["compute"] },
        lightIndices: { storage: d.arrayOf(d.u32), access: "readonly", visibility: ["compute"] },
        pointAtlas: { texture: d.textureDepth2d(), visibility: ["compute"] },
        pointShadows: { uniform: _pointCastersGpu, visibility: ["compute"] },
        shadowSamp: { sampler: "comparison", visibility: ["compute"] },
        shadowMap: { texture: d.textureDepth2d(), visibility: ["compute"] },
        sunShadow: { uniform: SunShadow, visibility: ["compute"] },
        lighting: { uniform: LightingGpu, visibility: ["compute"] },
        tileRects: { uniform: _tileRectsGpu, visibility: ["compute"] },
    })
    .$idx(1);

// `pointShadowRef()` is a plain host-side getter (its factory memoizes across the real-reference /
// raw-splice consumers, `sear/shade.ts`), not itself a TGSL value — calling it *inside* the kernel body
// reads as an untranspiled function to the resolver ("not marked with 'use gpu'"). Resolve it once here,
// in ordinary module-scope JS, so the kernel body below closes over the real `tgpu.fn` value directly.
const pointShadowOf = pointShadowRef();

/** the fog march compute kernel: per pixel, reconstruct the camera→fragment segment, then fuse the
 *  extinction march ({@link fogDensity} / `fogComposite`) with the clustered point/spot + sun in-scatter
 *  ({@link inScatterContribution} / {@link sunInScatter}), shadowed by sear's point atlas
 *  ({@link pointShadowOf}) and sun cascade map ({@link sampleSunShadow}). @internal */
export const fogKernel = tgpu
    .computeFn({
        workgroupSize: [WORKGROUP, WORKGROUP],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const dim = std.textureDimensions(fogLayout0.$.output);
        if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
        const px = d.vec2i(input.gid.xy);
        const scn = std.textureLoad(fogLayout0.$.sceneTex, px, 0).xyz;
        const depth = std.textureLoad(fogLayout0.$.depthTex, px, 0);

        // reconstruct the camera→fragment segment from the near plane to the fragment depth (reverse-Z:
        // near = ndc depth 1.0) — the same round-trip `reconstructWorld`'s own unit test pins
        const uv = std.div(std.add(d.vec2f(input.gid.xy), 0.5), d.vec2f(dim));
        const nearWorld = reconstructWorld(fogLayout0.$.view.invViewProj, uv, 1);
        const fragWorld = reconstructWorld(fogLayout0.$.view.invViewProj, uv, depth);
        const seg = std.sub(fragWorld, nearWorld);
        const dist = std.length(seg);
        const dir = std.div(seg, std.max(dist, 1e-6));

        // `cfg`/`camView` — not `fog`/`view` — to avoid colliding with the group-0 binding keys of the same
        // name (a local named identically to its own binding forces a confusing double-pointer alias)
        const cfg = fogLayout0.$.fog;
        const density = cfg.march.x;
        const base = cfg.march.y;
        const falloff = cfg.march.z;
        const jitter = cfg.march.w;
        const steps = std.max(d.u32(cfg.extra.x), d.u32(1));
        const g = cfg.extra.y;
        const absorption = cfg.extra.z;
        const gain = cfg.extra.w;
        const offset = std.mix(0.5, ign(d.vec2f(input.gid.xy)), jitter);

        // the froxel lookup along this pixel's ray: tile-xy is the pixel, the z-slice is the step's view
        // depth (matches sear's clusterOf for the same world point, perspective + ortho alike)
        const camView = fogLayout0.$.view;
        const near = camView.cluster.x;
        const far = camView.cluster.y;
        const forward = std.neg(std.cross(camView.right.xyz, camView.up.xyz));
        const slot = d.u32(camView.cluster.w);

        // the extinction + in-scatter march, fused on one front-to-back midpoint sweep — see march.ts's
        // header for the accumulator shapes (`trans` / `inScatter`)
        const ds = dist / d.f32(steps);
        const albedo = 1 - absorption;
        let trans = d.f32(1);
        let inScatter = d.vec3f(0);
        let i = d.u32(0);
        while (i < FOG_MAX_STEPS) {
            if (i >= steps) break;
            const p = std.add(nearWorld, std.mul(dir, (d.f32(i) + offset) * ds));
            const dens = fogDensity(p, density, base, falloff);
            const sampleTrans = std.exp(-dens * ds);
            const viewZ = std.max(std.dot(std.sub(p, camView.eye.xyz), forward), near);
            const entry = fogLayout1.$.lightGrid[clusterCell(uv.x, uv.y, viewZ, near, far, slot)];
            let lstep = d.vec3f(0);
            let j = d.u32(0);
            while (j < entry.y) {
                // the array-element read is a pointer, not a copy (3b-ii) — wrap in the element schema to
                // pass it by value into inScatterContribution / pointShadowRef
                const light = PointLightGpu(
                    fogLayout1.$.pointLights.lights[fogLayout1.$.lightIndices[entry.x + j]],
                );
                // params.x < 0 is the Volumetric flag; a plain light has no shaft (skip it) — the for-loop
                // `continue` becomes an early increment + `continue` under the dynamic-bound `while` shape
                // (3b-i's "loop{} emits as while(true)" class), never a reassociation
                if (light.params.x >= 0) {
                    j = j + 1;
                    continue;
                }
                const shadow = pointShadowOf(light, d.vec3f(0), p);
                lstep = d.vec3f(
                    std.add(lstep, std.mul(inScatterContribution(light, p, dir, g), shadow)),
                );
                j = j + 1;
            }
            // the sun (directional) shaft, additive with the clustered cones on the same accumulator
            if (fogLayout1.$.lighting.sunDirection.w > 0) {
                lstep = d.vec3f(
                    std.add(
                        lstep,
                        std.mul(
                            sunInScatter(
                                fogLayout1.$.lighting.sunColor.xyz,
                                fogLayout1.$.lighting.sunDirection.xyz,
                                dir,
                                g,
                            ),
                            sampleSunShadow(p, d.vec3f(0)),
                        ),
                    ),
                );
            }
            // (trans·albedo·gain) is one scalar product that scales lstep, and the result is scaled again
            // by (1−sampleTrans) as a separate step — matching the shipped shader's left-to-right f32
            // rounding; folding (1−sampleTrans) into the leading scalar chain shifts the last bits
            inScatter = d.vec3f(
                std.add(inScatter, std.mul(std.mul(lstep, trans * albedo * gain), 1 - sampleTrans)),
            );
            trans = trans * sampleTrans;
            i = i + 1;
        }

        // force the six shadow-only bindings into scope: `pointShadowRef()`/`sampleSunShadow`'s own bodies
        // read `pointAtlas`/`shadowSamp`/`pointShadows`/`tileRects`/`shadowMap`/`sunShadow` as free names,
        // invisible to the call-graph walk from here — folded (times zero) into the live `t` below rather
        // than a discarded local, so the JS→WGSL transpiler can't drop it as dead
        const forcedZero =
            (fogLayout1.$.pointShadows.casters[0].pos.x +
                fogLayout1.$.tileRects.rects[0].x +
                fogLayout1.$.sunShadow.enabled +
                std.textureSampleCompareLevel(
                    fogLayout1.$.pointAtlas,
                    fogLayout1.$.shadowSamp,
                    d.vec2f(0, 0),
                    0,
                ) +
                std.textureSampleCompareLevel(
                    fogLayout1.$.shadowMap,
                    fogLayout1.$.shadowSamp,
                    d.vec2f(0, 0),
                    0,
                )) *
            0;

        const t = trans + forcedZero;
        const outc = std.add(fogComposite(scn, cfg.color.xyz, t), inScatter);
        std.textureStore(fogLayout0.$.output, px, d.vec4f(outc, 1));
    })
    // "fogMarch", not "fog" — the kernel's own name shares one WGSL namespace with its bindings, and a
    // group-0 binding is keyed "fog" (the config uniform); naming both the same forces a confusing
    // collision-avoidance rename on the binding instead
    .$name("fogMarch");

/** the emitted fog-march WGSL — the device-free structural seam the fog test resolves. @internal */
export function fogWgsl(): string {
    return tgpu.resolve([fogKernel], { names: "strict" });
}
