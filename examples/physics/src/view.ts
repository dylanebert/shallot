import {
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
    qRotate,
    Shape,
    StandardMaterial,
    StandardRenderer,
    type System,
    sphere,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    drawLine,
    Gizmo,
    GizmoAsset,
    internText,
    LinesPlugin,
    Orbit,
    registerGizmo,
    Text,
} from "@dylanebert/shallot/extras";
import { Anchor, Billboard, worldToViewport } from "@dylanebert/shallot/rendering";
import { defaultDebugDraw, drawPhysics } from "@dylanebert/shallot/standard/physics";
import { EventSample } from "./events";
import { Grab } from "./grab";
import { scenePresentation, shapeMaterial, shapePresentation } from "./presentation";
import { castHeightField } from "./queries";
import { type Scene, scenes } from "./scenes";

const HIT_SEGMENT = { width: 2, depthBias: -1 };
const HIT_POINT = { width: 3, depthBias: -1 };

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
    if (scene === "heightField")
        addSystem({
            name: "height-field-casts",
            group: "draw",
            before: LinesPlugin.systems,
            update(world) {
                castHeightField(world);
                drawLine(world, [0, 0, 0], [0.4, 0, 0], 0xff0000);
                drawLine(world, [0, 0, 0], [0, 0.4, 0], 0x00ff00);
                drawLine(world, [0, 0, 0], [0, 0, 0.4], 0x0000ff);
            },
        });
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
        sensor: materials.add(presentation.materials.sensor),
    };
    const stateMaterials = new Map(
        Object.entries(roles).map(([role, handle]) => [
            presentation.materials[role as keyof typeof roles],
            handle,
        ]),
    );
    function materialFor(role: keyof typeof roles, eid: number) {
        const material = shapeMaterial(world, role, eid);
        let handle = stateMaterials.get(material);
        if (!handle) {
            handle = materials.add(material);
            stateMaterials.set(material, handle);
        }
        return handle;
    }
    const shapeRoles = new Map<EntityRef, keyof typeof roles>();
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
            const offset = qRotate(
                q.x.get(grab.body),
                q.y.get(grab.body),
                q.z.get(grab.body),
                q.w.get(grab.body),
                ax,
                ay,
                az,
            );
            const p = feedback.attachment;
            p[0] = pose.translation.x.get(grab.body) + offset[0];
            p[1] = pose.translation.y.get(grab.body) + offset[1];
            p[2] = pose.translation.z.get(grab.body) + offset[2];
            const transform = world.storage(Transform);
            transform.translation.set(feedback.marker, ...p);
            // Project a camera-up metre at the attachment depth to keep the diameter in CSS pixels.
            const cameraUp = qRotate(
                q.x.get(camera),
                q.y.get(camera),
                q.z.get(camera),
                q.w.get(camera),
                0,
                1,
                0,
            );
            const center = worldToViewport(world, camera, p);
            const up = worldToViewport(world, camera, [
                p[0] + cameraUp[0],
                p[1] + cameraUp[1],
                p[2] + cameraUp[2],
            ]);
            if (center && up) {
                const scale = 8 / Math.hypot(up[0] - center[0], up[1] - center[1]);
                transform.scale.set(feedback.marker, scale, scale, scale);
            }
            drawLine(world, p, grab.point, 0xeee9df);
        },
    });
    const meshes = world.resource(Meshes);
    let groundMesh: MeshHandle | undefined;
    if (presentation.ground) {
        const ground = world.create();
        groundMesh = meshes.add(presentation.ground.mesh);
        world.add(ground, Transform, { translation: presentation.ground.translation });
        world.add(ground, MeshInstance, { mesh: groundMesh });
        world.add(ground, MeshMaterial, roles.ground);
    }
    const grid = new GizmoAsset();
    for (const { a, b, color } of presentation.grid.lines) drawLine(grid, a, b, color);
    const gridEntity = world.create();
    world.add(gridEntity, Transform);
    world.add(gridEntity, Gizmo, {
        handle: registerGizmo(world, grid),
        width: presentation.grid.width,
        opacity: presentation.grid.opacity,
    });
    if (presentation.explosion.length) {
        const rings = world.create();
        const asset = new GizmoAsset();
        for (const { a, b, color } of presentation.explosion) drawLine(asset, a, b, color);
        world.add(rings, Transform);
        world.add(rings, Gizmo, { handle: registerGizmo(world, asset), width: 2 });
    }
    const views = new Map<EntityRef, MeshHandle>();
    const labels: number[] = [];
    function updateShapes(world: World): void {
        for (const [ref, mesh] of views) {
            const eid = world.resolve(ref);
            if (!eid || !world.has(eid, Shape)) {
                meshes.remove(mesh);
                views.delete(ref);
                shapeRoles.delete(ref);
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
            const role = shapeRoles.get(ref);
            if (role) {
                const handle = materialFor(role, eid);
                const storage = world.storage(MeshMaterial);
                if (
                    storage.material.get(eid) !== handle.material ||
                    storage.type.get(eid) !== handle.type
                ) {
                    storage.type.set(eid, handle.type);
                    storage.material.set(eid, handle.material);
                }
            }
            if (views.has(ref) && meshes.has(views.get(ref)!)) continue;
            const shape = initial.get(ref) ?? shapePresentation(world, scene, eid);
            initial.delete(ref);
            shapeRoles.set(ref, shape.role);
            const mesh = meshes.add(shape.mesh);
            views.set(ref, mesh);
            const edges = new GizmoAsset();
            for (const [a, b] of shape.edges) drawLine(edges, a, b, 0x000000);
            world.add(eid, Gizmo, {
                handle: registerGizmo(world, edges),
                width: 1,
                opacity: 0.45,
                depthBias: -0.0001,
            });
            world.add(eid, MeshInstance, { mesh });
            world.add(eid, MeshMaterial, materialFor(shape.role, eid));
        }
        for (let i = 0; i < sample.hits.length; i++) {
            const hit = sample.hits[i]!;
            const point: [number, number, number] = [hit.point.x, hit.point.y, hit.point.z];
            if (!labels[i]) {
                const label = world.create();
                labels[i] = label;
                world.add(label, Transform, {
                    translation: point,
                });
                world.add(label, Text, {
                    content: internText(
                        world,
                        `${hit.approachSpeed.toFixed(1)}, ${hit.userMaterialIdA}`,
                    ),
                    fontSize: 1,
                    depthBias: -1,
                    billboard: Billboard.Screen,
                    color: 0xeee9df,
                });
                world.add(label, Anchor, { offset: [...Anchor.BOTTOM_LEFT] });
            }
            drawLine(
                world,
                [
                    point[0] - hit.approachSpeed * hit.normal.x,
                    point[1] - hit.approachSpeed * hit.normal.y,
                    point[2] - hit.approachSpeed * hit.normal.z,
                ],
                point,
                0xffb62e,
                HIT_SEGMENT,
            );
            drawLine(
                world,
                [point[0] - 0.2, point[1], point[2]],
                [point[0] + 0.2, point[1], point[2]],
                0xffb62e,
                HIT_POINT,
            );
        }
    }
    updateShapes(world);
    addSystem({ name: "shape-views", group: "simulation", update: updateShapes });
    if (scene === "joint" || scene === "motor") {
        const draw = defaultDebugDraw();
        draw.drawJoints = true;
        draw.drawSegment = (a, b) => drawLine(world, [a.x, a.y, a.z], [b.x, b.y, b.z], 0x38a4b5);
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
        if (groundMesh !== undefined) meshes.remove(groundMesh);
        for (const material of stateMaterials.values()) materials.remove(material);
        materials.remove(markerMaterial);
        Object.assign(feedback, GrabFeedback.create());
    };
}
