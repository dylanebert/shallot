import { component, f32, u32, vec4 } from "../../engine";

/** Ambient illumination. color is hex sRGB; intensity is a linear multiplier. */
export const AmbientLight = component(
    "AmbientLight",
    { color: f32, intensity: f32 },
    {
        defaults: () => ({ color: 0xffffff, intensity: 0.5 }),
    },
);

/** Directional illumination. direction is the normalized travel direction when packed.
 * intensity remains a linear multiplier, not lux. shadowMapsEnabled is zero when disabled, one when enabled.
 * The shadow cascades follow Bevy's `CascadeShadowConfigBuilder`: numCascades depth slices (clamped to the
 * renderer's maximum) cover the camera's view out to maximumDistance in world units, the first ending at
 * firstCascadeFarBound and the rest spaced exponentially to maximumDistance; with one cascade the bound is
 * ignored and maximumDistance takes precedence. overlapProportion, in [0, 1), is the fraction of each
 * cascade blended with the next. All are read every frame; a numCascades change resizes the cascade atlas.
 * Shadow normal bias is measured in shadow texels; depth bias is a residual depth offset.
 */
export const DirectionalLight = component(
    "DirectionalLight",
    {
        color: f32,
        intensity: f32,
        direction: vec4,
        shadowMapsEnabled: u32,
        maximumDistance: f32,
        numCascades: u32,
        firstCascadeFarBound: f32,
        overlapProportion: f32,
        shadowDepthBias: f32,
        shadowNormalBias: f32,
    },
    {
        defaults: () => ({
            color: 0xffffff,
            intensity: 1.5,
            direction: [-0.6, -1.0, -0.8, 0],
            shadowMapsEnabled: 0,
            maximumDistance: 50,
            numCascades: 4,
            firstCascadeFarBound: 10,
            overlapProportion: 0.2,
            shadowDepthBias: 0.0005,
            shadowNormalBias: 1.8,
        }),
    },
);

/** Spherical light. Placement comes from GlobalTransform. color is hex sRGB;
 * intensity is a linear multiplier. range and radius are metres; falloff reaches zero at range.
 * shadowMapsEnabled is zero when disabled, one when enabled.
 * Shadow normal bias is measured in shadow texels; depth bias is a residual depth offset.
 */
export const PointLight = component(
    "PointLight",
    {
        color: f32,
        intensity: f32,
        range: f32,
        radius: f32,
        shadowMapsEnabled: u32,
        shadowDepthBias: f32,
        shadowNormalBias: f32,
    },
    {
        defaults: () => ({
            color: 0xffffff,
            intensity: 1,
            range: 10,
            radius: 0.1,
            shadowMapsEnabled: 0,
            shadowDepthBias: 0.0005,
            shadowNormalBias: 1.8,
        }),
    },
);

/** Cone light pointing down its GlobalTransform's local -Z. Units match PointLight;
 * Light attenuates smoothly between the inner and outer cones.
 * A SpotLight takes precedence over a PointLight on the same entity.
 */
export const SpotLight = component(
    "SpotLight",
    {
        color: f32,
        intensity: f32,
        range: f32,
        radius: f32,
        /** Inner half-angle in degrees, measured from the cone axis; full brightness inside it. */
        innerAngle: f32,
        /** Outer half-angle in degrees, measured from the cone axis; dark beyond it. */
        outerAngle: f32,
        shadowMapsEnabled: u32,
        shadowDepthBias: f32,
        shadowNormalBias: f32,
    },
    {
        defaults: () => ({
            color: 0xffffff,
            intensity: 1,
            range: 10,
            radius: 0.1,
            innerAngle: 20,
            outerAngle: 30,
            shadowMapsEnabled: 0,
            shadowDepthBias: 0.0005,
            shadowNormalBias: 1.8,
        }),
    },
);

/** Opt a light into FogPlugin's volumetric scattering. Without fog this marker is inert. */
export const VolumetricLight = component("VolumetricLight", {});

/** Exclude a mesh from shadow views without removing it from the main view. */
export const NotShadowCaster = component("NotShadowCaster", {});
