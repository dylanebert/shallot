import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    GlobalTransform,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    MotorJoint,
    NotShadowCaster,
    StandardMaterial,
    StandardRenderer,
    sphere,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { drawLine, Orbit } from "@dylanebert/shallot/extras";
import { worldToViewport } from "@dylanebert/shallot/rendering";
import { Grab } from "./grab";
import { scenePresentation } from "./presentation";
import { type Scene, scenes } from "./scenes";

export const GrabFeedback = {
    create: () => ({ marker: 0, attachment: [0, 0, 0] as [number, number, number] }),
};

export function createView(world: World, scene: Scene): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 380;
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 6000, shadowMapsEnabled: 1 });
    const camera = world.create();
    const view = scenes[scene].camera;
    world.add(camera, Camera, { clearColor: 0x191816 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, {
        yaw: (view.yaw * Math.PI) / 180,
        pitch: (view.pitch * Math.PI) / 180,
        distance: view.distance,
        pan: view.target,
    });
    world.add(camera, Transform);
    const materials = world.resource(Materials);
    const presentation = scenePresentation(world, scene);
    const roles = {
        ground: materials.add(presentation.materials.ground),
        subject: materials.add(presentation.materials.subject),
        motorized: materials.add(presentation.materials.motorized),
    };
    const markerMesh = world.resource(Meshes).add({ name: "Grab point", ...sphere(0.5, 16, 12) });
    const markerMaterial = materials.add(
        StandardMaterial({ baseColor: [0.855, 0.815, 0.738, 1], unlit: true }),
    );
    const feedback = world.resource(GrabFeedback);
    world.addSystem({
        name: "grab-feedback",
        group: "simulation",
        update(world) {
            const grab = world.resource(Grab);
            if (!grab.mouse) {
                if (feedback.marker) world.destroy(feedback.marker);
                feedback.marker = 0;
                return;
            }
            if (!feedback.marker) {
                feedback.marker = world.create();
                world.add(feedback.marker, Transform);
                world.add(feedback.marker, NotShadowCaster);
                world.add(feedback.marker, MeshInstance, { mesh: markerMesh });
                world.add(feedback.marker, MeshMaterial, markerMaterial);
            }
            const pose = world.storage(GlobalTransform);
            const anchor = world.storage(MotorJoint).localAnchorB;
            const ax = anchor.x.get(grab.joint),
                ay = anchor.y.get(grab.joint),
                az = anchor.z.get(grab.joint);
            const q = pose.rotation;
            const x = q.x.get(grab.body),
                y = q.y.get(grab.body),
                z = q.z.get(grab.body),
                w = q.w.get(grab.body);
            const tx = 2 * (y * az - z * ay),
                ty = 2 * (z * ax - x * az),
                tz = 2 * (x * ay - y * ax);
            const p = feedback.attachment;
            p[0] = pose.translation.x.get(grab.body) + ax + w * tx + y * tz - z * ty;
            p[1] = pose.translation.y.get(grab.body) + ay + w * ty + z * tx - x * tz;
            p[2] = pose.translation.z.get(grab.body) + az + w * tz + x * ty - y * tx;
            const transform = world.storage(Transform);
            transform.translation.set(feedback.marker, ...p);
            // Project a camera-up metre at the attachment depth to keep the diameter in CSS pixels.
            const cx = q.x.get(camera),
                cy = q.y.get(camera),
                cz = q.z.get(camera),
                cw = q.w.get(camera);
            const center = worldToViewport(world, camera, p);
            const up = worldToViewport(world, camera, [
                p[0] + 2 * (cx * cy - cw * cz),
                p[1] + 1 - 2 * (cx * cx + cz * cz),
                p[2] + 2 * (cy * cz + cw * cx),
            ]);
            if (center && up) {
                const scale = 8 / Math.hypot(up[0] - center[0], up[1] - center[1]);
                transform.scale.set(feedback.marker, scale, scale, scale);
            }
            drawLine(world, p, grab.point, 0xeee9df, 1);
        },
    });
    const meshes = world.resource(Meshes);
    for (const { eid, body, role, mesh } of presentation.shapes) {
        if (body !== eid) world.add(eid, Transform, world.get(body, Transform)!);
        world.add(eid, MeshInstance, { mesh: meshes.add(mesh) });
        world.add(eid, MeshMaterial, roles[role]);
    }
}
