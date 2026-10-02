import { f32, u32, vec4 } from "../../engine";

/** Ambient illumination. color is hex sRGB; intensity is a linear multiplier. */
export const AmbientLight = { color: f32, intensity: f32 };

/** Directional illumination. direction is the normalized travel direction when packed.
 * intensity remains a linear multiplier, not lux. maximumDistance bounds the shadow cascades in world units.
 * shadowMapsEnabled is zero when disabled, one when enabled.
 * Shadow normal bias is measured in shadow texels; depth bias is a residual depth offset.
 */
export const DirectionalLight = {
    color: f32,
    intensity: f32,
    direction: vec4,
    shadowMapsEnabled: u32,
    maximumDistance: f32,
    shadowDepthBias: f32,
    shadowNormalBias: f32,
};

/** Spherical light. Placement comes from GlobalTransform. color is hex sRGB;
 * intensity is a linear multiplier. range and radius are metres; falloff reaches zero at range.
 * shadowMapsEnabled is zero when disabled, one when enabled.
 * Shadow normal bias is measured in shadow texels; depth bias is a residual depth offset.
 */
export const PointLight = {
    color: f32,
    intensity: f32,
    range: f32,
    radius: f32,
    shadowMapsEnabled: u32,
    shadowDepthBias: f32,
    shadowNormalBias: f32,
};

/** Cone light pointing down its GlobalTransform's local -Z. Units match PointLight;
 * innerAngle and outerAngle are half-angles in radians, with smooth attenuation between them.
 * A SpotLight takes precedence over a PointLight on the same entity.
 */
export const SpotLight = {
    color: f32,
    intensity: f32,
    range: f32,
    radius: f32,
    innerAngle: f32,
    outerAngle: f32,
    shadowMapsEnabled: u32,
    shadowDepthBias: f32,
    shadowNormalBias: f32,
};

/** Opt a light into FogPlugin's volumetric scattering. Without fog this marker is inert. */
export const VolumetricLight = {};

/** Exclude a mesh from shadow views without removing it from the main view. */
export const NotShadowCaster = {};
