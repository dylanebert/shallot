import { f32, u8 } from "../../engine";

// Membership marks the initial pose snap. Derived display state is not authored or serialized.
export const OrbitSmooth = {
    yaw: f32,
    pitch: f32,
    distance: f32,
    size: f32,
    keyYawVelocity: f32,
    keyPitchVelocity: f32,
    flyActive: u8,
};
