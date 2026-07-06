import type { State } from "../../engine";
import { Compute, f32, sparse, unpackColor, vec4 } from "../../engine";
import { slab } from "../slab";
import { Transform } from "../transforms";

/**
 * ambient light component. sear's `lit` / `lightFactor` helpers
 * read it via the kitchen Lighting uniform. `color` is hex sRGB (e.g.
 * `0xd0dcec`); `intensity` is a linear multiplier
 *
 * @example
 * ```
 * <a ambient-light="color: 0xd0dcec; intensity: 1.13" />
 * ```
 */
export const AmbientLight = {
    color: sparse(f32),
    intensity: sparse(f32),
};

/**
 * directional light component. sear's `lit` / `lightFactor`
 * helpers read it via the kitchen Lighting uniform. `direction` is the light's
 * travel direction (down-pointing for a sun overhead); auto-normalized when
 * packed
 *
 * @example
 * ```
 * <a directional-light="direction: -0.3 -0.8 -0.55; color: 0xfff4e0; intensity: 1.2" />
 * ```
 */
export const DirectionalLight = {
    color: sparse(f32),
    intensity: sparse(f32),
    direction: sparse(vec4),
};

/**
 * point light component. Position comes from the entity's `Transform`; sear's
 * `lit` / `lightFactor` helpers accumulate the fragment's cluster's point
 * lights: inverse-square falloff windowed smoothly to exactly zero at
 * `range`. `color` is hex sRGB; `intensity` is a linear multiplier. Slab
 * fields: the light-cull compute passes read them straight off the GPU (no
 * CPU light list)
 *
 * @example
 * ```
 * <a point-light="color: 0xffd9a0; intensity: 2; range: 6" transform="pos: 0 1.8 0" />
 * ```
 */
export const PointLight = {
    /** the light's hex sRGB color (e.g. 0xffd9a0) */
    color: slab(f32),
    /** linear brightness multiplier */
    intensity: slab(f32),
    /** the distance (metres) the falloff smoothly reaches zero at: the cull cutoff */
    range: slab(f32),
    /** the physical source radius (metres): a soft sphere, not a point. Larger softens the near-field
     * bulb and widens the specular highlight; 0.01 reproduces the old bare-filament hotspot */
    radius: slab(f32),
};

/**
 * spot add-on for a {@link PointLight}: presence narrows the light into a cone (like {@link Shadow},
 * presence is the switch). The cone points along the entity's forward axis (its `Transform` rotation), so
 * aim it by rotating the entity. `inner` / `outer` are half-angles in degrees (axis to edge): full
 * brightness inside `inner`, smoothly to dark at `outer`. The light still falls off + culls by the
 * PointLight's `range`.
 *
 * @example
 * ```
 * <a point-light="color: 0xffffff; intensity: 4; range: 12" spot="inner: 18; outer: 28" transform="rot: -45 0 0" />
 * ```
 */
export const Spot = {
    /** the cone's inner half-angle (degrees, axis→edge): full brightness inside it */
    inner: slab(f32),
    /** the cone's outer half-angle (degrees, axis→edge): dark past it, smooth between inner and outer */
    outer: slab(f32),
};

/**
 * opt a light ({@link PointLight}, {@link Spot}, or the {@link DirectionalLight} sun) into volumetric
 * light shafts; presence is the switch, like {@link Spot}. On a point/spot light the light-compact pass
 * flags its compacted entry; on the sun it flags the Lighting uniform. The `fog` march then scatters that
 * light through the haze (a visible cone or sun shaft, shadowed by occluders if the light also carries a
 * `Shadow`). With no `FogPlugin` / `Fog` singleton the flag is inert. The lit path is unchanged.
 *
 * @example
 * ```
 * <a point-light="color: 0xffffff; intensity: 6; range: 14" spot="inner: 16; outer: 26" volumetric transform="pos: 0 8 0; rot: -90 0 0" />
 * <a directional-light="direction: -0.4 -0.8 -0.45" volumetric shadow="distance: 80" />
 * ```
 */
export const Volumetric = {};

/** the Lighting UBO byte size (three vec4s: ambient, sun direction, sun color); a relocatable consumer
 * (the fog march) sizes its `lighting` binding to match. */
export const LIGHTING_UNIFORM_SIZE = 48;

/** the `Lighting` UBO's WGSL struct, spliced by sear + any relocatable consumer (the fog march) that
 * reads the shared lighting uniform; layout mirrors {@link Lighting}. */
export const LIGHTING_STRUCT_WGSL = /* wgsl */ `
struct Lighting {
    ambientColor: vec4<f32>,
    sunDirection: vec4<f32>,
    sunColor: vec4<f32>,
}`;

/**
 * GPU Lighting UBO + CPU staging mirror, written once per frame by
 * {@link writeLighting}. Layout matches {@link LIGHTING_STRUCT_WGSL}:
 * `ambientColor.rgb` is linear color, `ambientColor.a` is intensity (shader
 * multiplies); `sunColor.rgb` has intensity pre-baked; `sunDirection.xyz` is
 * the normalized travel direction (light-to-surface)
 * @expand
 */
export interface Lighting {
    buffer: GPUBuffer;
    staging: Float32Array;
}

const _backing = new ArrayBuffer(LIGHTING_UNIFORM_SIZE);

export const Lighting: Lighting = {
    buffer: null!,
    staging: new Float32Array(_backing),
};

/** read the singleton AmbientLight + DirectionalLight entities and pack the Lighting UBO */
export function writeLighting(state: State): void {
    if (!Compute.device || !Lighting.buffer) return;

    // zero first; absent lights + pad lanes stay 0, so each present light just
    // writes its own fields (no sun → sunColor 0 → no contribution either way)
    const s = Lighting.staging;
    s.fill(0);

    const ambient = state.only([AmbientLight]);
    if (ambient >= 0) {
        const rgb = unpackColor(AmbientLight.color.get(ambient));
        s[0] = rgb.r;
        s[1] = rgb.g;
        s[2] = rgb.b;
        s[3] = AmbientLight.intensity.get(ambient);
    }

    const dir = state.only([DirectionalLight]);
    if (dir >= 0) {
        const dx = DirectionalLight.direction.x.get(dir);
        const dy = DirectionalLight.direction.y.get(dir);
        const dz = DirectionalLight.direction.z.get(dir);
        const len = Math.hypot(dx, dy, dz);
        if (len < 1e-4) {
            s[5] = -1; // degenerate direction → straight down
        } else {
            s[4] = dx / len;
            s[5] = dy / len;
            s[6] = dz / len;
        }
        const rgb = unpackColor(DirectionalLight.color.get(dir));
        const i = DirectionalLight.intensity.get(dir);
        s[8] = rgb.r * i;
        s[9] = rgb.g * i;
        s[10] = rgb.b * i;
        // the sun's volumetric opt-in: a `Volumetric` marker flags the otherwise-pad sunDirection.w lane
        // (1 = scatter shafts in the fog march). The lit path reads only sunDirection.xyz, so the flag is
        // inert there — the analogue of the point light's radius-sign flag, no 4th vec4
        if (state.has(dir, Volumetric)) s[7] = 1;
    }

    Compute.device.queue.writeBuffer(Lighting.buffer, 0, s as Float32Array<ArrayBuffer>);
}

/** the point-light list cap. The compacted list the cull pass bins is fixed-size so sear's binding
 * exists for every surface; overflow warns, never silently truncates. */
export const MAX_POINT_LIGHTS = 256;

const POINT_LIGHT_FLOATS = 12; // posRange vec4 + color vec4 + params vec4
const POINT_HEADER_FLOATS = 4; // count, padded to 16B
export const POINT_LIGHTS_BUFFER_SIZE =
    (POINT_HEADER_FLOATS + MAX_POINT_LIGHTS * POINT_LIGHT_FLOATS) * 4;

/**
 * the compacted point-light list's WGSL struct (`PointLightGpu[]` + count header), spliced by sear's
 * clustered loop and the fog march. Per light: `posRange` (xyz world position, w = 1/range²), `color`
 * (linear rgb intensity-baked, `a` = source entity id as f32: the per-entity hook sear matches shadowed
 * casters on), `params` (x = source radius; y = the spot cone axis oct-packed via bitcast; z/w = the
 * Frostbite spot angular scale/offset: a non-spot writes `(radius, 0, 0, 1)` so the angular factor is 1).
 * GPU-written by the light compact pass (`cluster.ts`) from the PointLight + Spot slabs + the transforms
 * firehose. There is no CPU light list.
 */
export const POINT_LIGHTS_STRUCT_WGSL = /* wgsl */ `
struct PointLightGpu {
    posRange: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
}
struct PointLights {
    count: vec4<u32>,
    lights: array<PointLightGpu>,
}`;

/**
 * the point-light falloff (Bevy `getDistanceAttenuation`): inverse-square with a smooth
 * window (`smooth = saturate(1 − (d²/r²)²)`, attenuation `smooth² / max(d², radiusSq)`),
 * exactly zero at and past the range, and flat at `1/radiusSq` inside the source sphere
 * (Karis representative point: `radiusSq = 0` would spike toward ∞ at the bulb). Pure; the
 * oracle sear's WGSL twin is pinned to
 */
export function distanceAttenuation(distSq: number, invRangeSq: number, radiusSq: number): number {
    const factor = distSq * invRangeSq;
    const smooth = Math.min(Math.max(1 - factor * factor, 0), 1);
    return (smooth * smooth) / Math.max(distSq, radiusSq);
}

/**
 * the spot cone's angular-attenuation coefficients (Frostbite `getAngleAtt`, Lagarde 2014) from the
 * inner/outer half-angles (degrees). The FS multiplies the light by `saturate(cd·scale + offset)²`, cd =
 * cos(angle between the cone axis and the light→fragment direction): 1 inside the inner cone, smoothly to
 * 0 at the outer. The GPU compact pass bakes these into the light's `params.zw`; the oracle its WGSL twin
 * is pinned to. `inner == outer` is a hard edge (the divide is clamped), never a NaN.
 */
export function spotParams(innerDeg: number, outerDeg: number): { scale: number; offset: number } {
    const cosInner = Math.cos((innerDeg * Math.PI) / 180);
    const cosOuter = Math.cos((outerDeg * Math.PI) / 180);
    const scale = 1 / Math.max(cosInner - cosOuter, 1e-4);
    return { scale, offset: -cosOuter * scale };
}

let _overflowWarned = false;

/**
 * warn once per episode when more PointLight entities exist than the list cap:
 * the GPU compact pass drops the excess (count beyond {@link MAX_POINT_LIGHTS}
 * never writes an entry), so the overflow is loud, not silent. A count, not a
 * pack: the light data itself flows GPU-side
 */
export function warnLightOverflow(state: State): void {
    let count = 0;
    for (const _ of state.query([PointLight, Transform])) count++;
    if (count > MAX_POINT_LIGHTS) {
        if (!_overflowWarned) {
            _overflowWarned = true;
            console.warn(
                `kitchen: ${count} point lights exceed the ${MAX_POINT_LIGHTS} cap; ${count - MAX_POINT_LIGHTS} ignored`,
            );
        }
    } else {
        _overflowWarned = false;
    }
}
