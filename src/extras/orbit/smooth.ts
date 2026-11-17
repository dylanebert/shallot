import { entity, f32, u8, vec3 } from "../../engine";

// Membership marks the initial pose snap. Derived display state is not authored or serialized.
export const OrbitSmooth = {
    yaw: f32,
    pitch: f32,
    distance: f32,
    size: f32,
    keyYawVelocity: f32,
    keyPitchVelocity: f32,
    flyActive: u8,
    panImpulse: vec3,
    pan: vec3,
    panTarget: entity,
    touchStopped: u8,
    zoomDistanceImpulse: f32,
    zoomSizeImpulse: f32,
    zoomDistance: f32,
    zoomSize: f32,
    publishedYaw: f32,
    publishedPitch: f32,
    homeYaw: f32,
    homePitch: f32,
    homeDistance: f32,
    homeSize: f32,
    homePan: vec3,
};
