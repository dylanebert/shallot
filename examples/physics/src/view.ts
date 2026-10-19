import {
    AlphaMode,
    Camera,
    DirectionalLight,
    type EntityRef,
    GlobalAmbientLight,
    GlobalTransform,
    lookAtRotation,
    Materials,
    Meshes,
    type MeshHandle,
    MeshInstance,
    MeshMaterial,
    MotorJoint,
    NotShadowCaster,
    Shape,
    StandardMaterial,
    StandardRenderer,
    type System,
    sphere,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { drawLine, internText, Orbit, Text } from "@dylanebert/shallot/extras";
import { Anchor, Billboard, worldToViewport } from "@dylanebert/shallot/rendering";
import { defaultDebugDraw, drawPhysics } from "@dylanebert/shallot/standard/physics";
import { EventSample } from "./events";
import { Grab } from "./grab";
import { scenePresentation, shapePresentation } from "./presentation";
import { type Scene, scenes } from "./scenes";

export const GrabFeedback = {
    create: () => ({ marker: 0, attachment: [0, 0, 0] as [number, number, number] }),
};

export function createViewer(world: World): (world: World, scene: Scene) => () => void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 380;
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 6000, shadowMapsEnabled: 1 });
    const camera = world.create();
    world.add(camera, Camera, { clearColor: 0x191816 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit);
    world.add(camera, Transform);
    return (world, scene) => createView(world, scene, camera);
}

function createView(world: World, scene: Scene, camera: number): () => void {
    const systems: System[] = [];
    function addSystem(system: System): void {
        systems.push(system);
        world.addSystem(system);
    }
    const view = scenes[scene].camera;
    const orbit = world.storage(Orbit);
    orbit.yaw.set(camera, (view.yaw * Math.PI) / 180);
    orbit.pitch.set(camera, (view.pitch * Math.PI) / 180);
    orbit.distance.set(camera, view.distance);
    orbit.pan.set(camera, view.target[0]!, view.target[1]!, view.target[2]!);
    const materials = world.resource(Materials);
    const presentation = scenePresentation(world, scene);
    const initial = new Map(presentation.shapes.map((shape) => [world.ref(shape.eid), shape]));
    const roles = {
        ground: materials.add(presentation.materials.ground),
        subject: materials.add(presentation.materials.subject),
        motorized: materials.add(presentation.materials.motorized),
        sensor: materials.add(presentation.materials.sensor, { alphaMode: AlphaMode.Blend }),
    };
    const markerMesh = world.resource(Meshes).add({ name: "Grab point", ...sphere(0.5, 16, 12) });
    const markerMaterial = materials.add(
        StandardMaterial({ baseColor: [0.855, 0.815, 0.738, 1], unlit: true }),
    );
    const feedback = world.resource(GrabFeedback);
    addSystem({
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
    const views = new Map<EntityRef, MeshHandle>();
    const labels: number[] = [];
    function updateShapes(world: World): void {
        for (const [ref, mesh] of views) {
            const eid = world.resolve(ref);
            if (!eid || !world.has(eid, Shape)) {
                meshes.remove(mesh);
                views.delete(ref);
            }
        }
        const sample = world.resource(EventSample);
        for (const eid of world.query([Shape])) {
            const body = world.storage(Shape).body.get(eid) || eid;
            if (body !== eid) {
                const pose = world.get(body, Transform);
                if (pose) {
                    if (!world.has(eid, Transform)) world.add(eid, Transform, pose);
                    else {
                        const transform = world.storage(Transform);
                        transform.translation.set(eid, ...pose.translation);
                        transform.rotation.set(eid, ...pose.rotation);
                        transform.scale.set(eid, ...pose.scale);
                    }
                }
            }
            const ref = world.ref(eid);
            if (views.has(ref) && meshes.has(views.get(ref)!)) continue;
            const shape = initial.get(ref) ?? shapePresentation(world, scene, eid);
            initial.delete(ref);
            const mesh = meshes.add(shape.mesh);
            views.set(ref, mesh);
            world.add(eid, MeshInstance, { mesh });
            world.add(eid, MeshMaterial, roles[shape.role]);
        }
        for (let i = 0; i < sample.hits.length; i++) {
            const hit = sample.hits[i]!;
            const point: [number, number, number] = [
                hit.point.x + 0.05 * hit.normal.x,
                hit.point.y + 0.05 * hit.normal.y,
                hit.point.z + 0.05 * hit.normal.z,
            ];
            if (!labels[i]) {
                const label = world.create();
                labels[i] = label;
                world.add(label, Transform, {
                    translation: [
                        point[0] + hit.normal.x,
                        point[1] + hit.normal.y,
                        point[2] + hit.normal.z,
                    ],
                });
                world.add(label, Text, {
                    content: internText(
                        world,
                        `${hit.approachSpeed.toFixed(1)}, ${hit.userMaterialIdA}`,
                    ),
                    fontSize: 1,
                    billboard: Billboard.Screen,
                    color: 0xeee9df,
                });
                world.add(label, Anchor, { offset: [...Anchor.BOTTOM_LEFT] });
            }
            // Incoming direction and speed from the hit record, placed above the opaque ground
            // rather than below it: Shallot's lines/text are depth-tested, not debug overlays.
            drawLine(
                world,
                [
                    point[0] + hit.approachSpeed * hit.normal.x,
                    point[1] + hit.approachSpeed * hit.normal.y,
                    point[2] + hit.approachSpeed * hit.normal.z,
                ],
                point,
                0xffb62e,
                2,
            );
            drawLine(
                world,
                [point[0] - 0.2, point[1], point[2]],
                [point[0] + 0.2, point[1], point[2]],
                0xffb62e,
                3,
            );
        }
    }
    updateShapes(world);
    if (scene === "contact" || scene === "sensor" || scene === "hit")
        addSystem({ name: "shape-views", group: "simulation", update: updateShapes });
    if (scene === "joint" || scene === "motor") {
        const draw = defaultDebugDraw();
        draw.drawJoints = true;
        draw.drawSegment = (a, b) => drawLine(world, [a.x, a.y, a.z], [b.x, b.y, b.z], 0x38a4b5, 1);
        addSystem({
            name: "joint-views",
            group: "simulation",
            update(world) {
                drawPhysics(world, draw);
            },
        });
    }
    return () => {
        for (const system of systems) world.removeSystem(system);
        for (const mesh of views.values()) meshes.remove(mesh);
        meshes.remove(markerMesh);
        for (const material of Object.values(roles)) materials.remove(material);
        materials.remove(markerMaterial);
        Object.assign(feedback, GrabFeedback.create());
    };
}
