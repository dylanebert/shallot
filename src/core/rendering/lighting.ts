import { f32, vec4 } from "../../engine";

/**
 * ambient light component. sear's `lit` / `lightFactor` helpers
 * read it via the shallot Lighting uniform. `color` is hex sRGB (e.g.
 * `0xd0dcec`); `intensity` is a linear multiplier
 *
 * @example
 * ```
 * world.add(world.create(), AmbientLight, { color: 0xd0dcec, intensity: 1.13 });
 * ```
 */
export const AmbientLight = {
    color: f32,
    intensity: f32,
};

/**
 * directional light component. sear's `lit` / `lightFactor`
 * helpers read it via the shallot Lighting uniform. `direction` is the light's
 * travel direction (down-pointing for a sun overhead); auto-normalized when
 * packed
 *
 * @example
 * ```
 * world.add(world.create(), DirectionalLight, {
 *     direction: [-0.3, -0.8, -0.55, 0],
 *     color: 0xfff4e0,
 *     intensity: 1.2,
 * });
 * ```
 */
export const DirectionalLight = {
    color: f32,
    intensity: f32,
    direction: vec4,
};

/**
 * point light component. Position comes from the entity's `Transform`; sear's
 * `lit` / `lightFactor` helpers accumulate the fragment's cluster's point
 * lights: inverse-square falloff windowed smoothly to exactly zero at
 * `range`. `color` is hex sRGB; `intensity` is a linear multiplier. Dense table
 * fields: the light-cull compute pass reads them from struct records (no
 * CPU light list)
 *
 * @example
 * ```
 * const lamp = world.create();
 * world.add(lamp, PointLight, { color: 0xffd9a0, intensity: 2, range: 6 });
 * world.add(lamp, Transform, { translation: [0, 1.8, 0, 0] });
 * ```
 */
export const PointLight = {
    /** the light's hex sRGB color (e.g. 0xffd9a0) */
    color: f32,
    /** linear brightness multiplier */
    intensity: f32,
    /** the distance (metres) the falloff smoothly reaches zero at: the cull cutoff */
    range: f32,
    /** the physical source radius (metres): a soft sphere, not a point. Larger softens the near-field
     * bulb and widens the specular highlight; 0.01 reproduces the old bare-filament hotspot */
    radius: f32,
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
 * const spot = world.create();
 * world.add(spot, PointLight, { color: 0xffffff, intensity: 4, range: 12 });
 * world.add(spot, Spot, { inner: 18, outer: 28 });
 * const q = eulerToQuat(-45, 0, 0);
 * world.add(spot, Transform, { rotation: [q.x, q.y, q.z, q.w] });
 * ```
 */
export const Spot = {
    /** the cone's inner half-angle (degrees, axis→edge): full brightness inside it */
    inner: f32,
    /** the cone's outer half-angle (degrees, axis→edge): dark past it, smooth between inner and outer */
    outer: f32,
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
 * const spot = world.create();
 * world.add(spot, PointLight, { color: 0xffffff, intensity: 6, range: 14 });
 * world.add(spot, Spot, { inner: 16, outer: 26 });
 * world.add(spot, Volumetric);
 * const q = eulerToQuat(-90, 0, 0);
 * world.add(spot, Transform, { translation: [0, 8, 0, 0], rotation: [q.x, q.y, q.z, q.w] });
 * const sun = world.create();
 * world.add(sun, DirectionalLight, { direction: [-0.4, -0.8, -0.45, 0] });
 * world.add(sun, Volumetric);
 * world.add(sun, Shadow, { distance: 80 });
 * ```
 */
export const Volumetric = {};
