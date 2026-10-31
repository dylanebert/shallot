import { Camera, type World } from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import type { Gltf } from "@dylanebert/shallot/gltf";
import { type Scene, truckPlacement } from "./scenes";

// three.js examples/webgl_loader_gltf.html uses a 1.3 fit offset. Express that
// margin as projected sphere coverage, fitting the narrower axis even on a phone.
export const frameShare = 1 / 1.3;

export function sceneBounds(model: Gltf, scene: Scene): Gltf["bounds"] {
    const placement = scene === "CesiumMilkTruck" ? truckPlacement : {};
    const min = model.bounds.min.map(
        (value, axis) =>
            value * (placement.scale?.[axis] ?? 1) + (placement.translation?.[axis] ?? 0),
    ) as [number, number, number];
    const max = model.bounds.max.map(
        (value, axis) =>
            value * (placement.scale?.[axis] ?? 1) + (placement.translation?.[axis] ?? 0),
    ) as [number, number, number];
    return { min, max };
}

export function fitBounds(world: World, camera: number, bounds: Gltf["bounds"], aspect: number) {
    const center = bounds.min.map((value, axis) => (value + bounds.max[axis]) / 2);
    const radius = Math.hypot(...bounds.max.map((value, axis) => value - center[axis]));
    const cameras = world.storage(Camera);
    const halfFov = (cameras.fov.get(camera) * Math.PI) / 360;
    const tangent = Math.tan(halfFov) * Math.min(1, aspect);
    // A sphere's silhouette has tan(angle) = r / sqrt(distance² - r²).
    const distance = radius * Math.sqrt(1 + 1 / (frameShare * tangent) ** 2);
    const orbit = world.storage(Orbit);
    orbit.pan.set(camera, center[0], center[1], center[2]);
    orbit.distance.set(camera, distance);
    orbit.minDistance.set(camera, radius / 10);
    orbit.maxDistance.set(camera, distance * 10);
    cameras.near.set(camera, distance / 100);
    cameras.far.set(camera, distance * 100);
}

export function sceneAngles(scene: Scene) {
    // Sample Renderer UserCamera.resetView starts level and front-on.
    if (scene === "TwoSidedPlane") {
        // Its normal is +Y: start below the sheet so the double-sided underside is visible.
        return { yaw: 0, pitch: -Math.PI / 6 };
    }
    if (scene === "NormalTangentMirrorTest") {
        // Its tangent comparison faces +Z; keep that lit face square to the camera.
        return { yaw: 0, pitch: 0 };
    }
    return { yaw: 0, pitch: 0 };
}
