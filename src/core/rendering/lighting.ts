import type { Resource } from "../../engine";
import { component, f32, u32 } from "../../engine";
import { Transform } from "../transform";

/** Ambient illumination used unless a camera carries an {@link AmbientLight} override. */
export interface GlobalAmbientLightState {
    /** Linearized from this sRGB hex color when packed. */
    color: number;
    /** Scene luminance in cd/m². */
    brightness: number;
}

/** The world's default ambient source, measured in cd/m². */
export const GlobalAmbientLight: Resource<GlobalAmbientLightState> = {
    create: () => ({ color: 0xffffff, brightness: 80 }),
};

/** Per-camera ambient override. Color is sRGB hex; brightness is luminance in cd/m². */
export const AmbientLight = component(
    "AmbientLight",
    { color: f32, brightness: f32 },
    { defaults: () => ({ color: 0xffffff, brightness: 80 }) },
);

/** Directional illumination in lux. Light travels along the entity's transformed local -Z. */
export const DirectionalLight = component(
    "DirectionalLight",
    {
        color: f32,
        illuminance: f32,
        shadowMapsEnabled: u32,
        maximumDistance: f32,
        numCascades: u32,
        firstCascadeFarBound: f32,
        overlapProportion: f32,
        /** Receiver offset toward this light in world units. */
        depthBias: f32,
        /** Receiver offset in shadow texels along the surface normal. */
        shadowNormalBias: f32,
    },
    {
        defaults: () => ({
            color: 0xffffff,
            illuminance: 4703.4155,
            shadowMapsEnabled: 0,
            maximumDistance: 50,
            numCascades: 4,
            firstCascadeFarBound: 10,
            overlapProportion: 0.2,
            depthBias: 0.0005,
            shadowNormalBias: 1.8,
        }),
        requires: [Transform],
    },
);

/** Spherical source with luminous flux in lumens. Placement, range and radius use world units. */
export const PointLight = component(
    "PointLight",
    {
        color: f32,
        /** Total emitted luminous flux in lumens. */
        intensity: f32,
        range: f32,
        radius: f32,
        shadowMapsEnabled: u32,
        /** Receiver offset toward this light in world units. */
        depthBias: f32,
        shadowNormalBias: f32,
    },
    {
        defaults: () => ({
            color: 0xffffff,
            intensity: 39_403.24,
            range: 10,
            radius: 0.1,
            shadowMapsEnabled: 0,
            depthBias: 0.0005,
            shadowNormalBias: 1.8,
        }),
        requires: [Transform],
    },
);

/** Cone source with luminous flux in lumens. It points down GlobalTransform's local -Z. */
export const SpotLight = component(
    "SpotLight",
    {
        color: f32,
        /** Total emitted luminous flux in lumens. */
        intensity: f32,
        range: f32,
        radius: f32,
        /** Inner half-angle in degrees; full brightness inside it. */
        innerAngle: f32,
        /** Outer half-angle in degrees; dark beyond it. */
        outerAngle: f32,
        shadowMapsEnabled: u32,
        /** Receiver offset toward this light in world units. */
        depthBias: f32,
        shadowNormalBias: f32,
    },
    {
        defaults: () => ({
            color: 0xffffff,
            intensity: 39_403.24,
            range: 10,
            radius: 0.1,
            innerAngle: 20,
            outerAngle: 30,
            shadowMapsEnabled: 0,
            depthBias: 0.0005,
            shadowNormalBias: 1.8,
        }),
        requires: [Transform],
    },
);

/** Bevy's apparent solar diameter used by `SunDisk.EARTH` (radians). */
export const SUN_DISK_EARTH_ANGULAR_SIZE = 0.00930842;

/** Visible disk attached to a directional light. Sky's angular size and glow are independent of illumination. */
export const SunDisk = component(
    "SunDisk",
    {
        /** Apparent disk diameter in radians. */
        angularSize: f32,
        /** Visual multiplier; does not change the light or its shadows. */
        intensity: f32,
        /** Per-light sky glow, a Shallot extension to Bevy's SunDisk. */
        glow: f32,
    },
    {
        defaults: () => ({ angularSize: SUN_DISK_EARTH_ANGULAR_SIZE, intensity: 1, glow: 0 }),
        requires: [DirectionalLight],
    },
);

/** Opt a light into FogPlugin's volumetric scattering. Without fog this marker is inert. */
export const VolumetricLight = component("VolumetricLight", {});

/** Exclude a mesh from shadow views without removing it from the main view. */
export const NotShadowCaster = component("NotShadowCaster", {});
