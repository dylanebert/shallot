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
    OrbitPlugin,
    registerGizmo,
    Text,
} from "@dylanebert/shallot/extras";
import { Anchor, Billboard, worldToViewport } from "@dylanebert/shallot/rendering";
import { defaultDebugDraw, drawPhysics } from "@dylanebert/shallot/standard/physics";
import { drawingBounds, drawSet } from "./draw-set";
import { DrivingState } from "./driving";
import { EventSample } from "./events";
import { Grab } from "./grab";
import { drawJointViews } from "./joint-presentation";
import {
    drivenBodies,
    type ShapeGeometryCache,
    scenePresentation,
    shapeMaterial,
    shapePresentation,
    shapeRole,
} from "./presentation";
import { castHeightField } from "./queries";
import { type Scene, scenes } from "./scenes";

function placeMarker(
    world: World,
    camera: number,
    marker: number,
    point: [number, number, number],
    diameter: number,
): void {
    const transform = world.storage(Transform);
    transform.translation.set(marker, ...point);
    const q = world.storage(GlobalTransform).rotation;
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
    const center = worldToViewport(world, camera, point);
    const up = worldToViewport(world, camera, [
        point[0] + cameraUp[0],
        point[1] + cameraUp[1],
        point[2] + cameraUp[2],
    ]);
    if (center && up) {
        const scale = diameter / Math.hypot(up[0] - center[0], up[1] - center[1]);
        transform.scale.set(marker, scale, scale, scale);
    }
}

import { drawEventMark, drawSensorHitViews } from "./event-presentation";

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

export function setSceneCamera(world: World, scene: Scene, camera: number): void {
    const view = scenes[scene].camera;
    const orbit = world.storage(Orbit);
    orbit.yaw.set(camera, (view.yaw * Math.PI) / 180);
    orbit.pitch.set(camera, (view.pitch * Math.PI) / 180);
    const phone = typeof window !== "undefined" && window.innerWidth <= 480;
    orbit.distance.set(
        camera,
        phone && "phoneDistance" in view ? view.phoneDistance : view.distance,
    );
    orbit.target.set(camera, scene === "driving" ? world.resource(DrivingState).chassis : 0);
    // Sharing the orbit button prevents flythrough from detaching the driving camera.
    orbit.flyButton.set(camera, scene === "driving" ? orbit.orbitButton.get(camera) : 2);
    const target = scene === "driving" ? [0, 0, 0] : view.target;
    orbit.pan.set(camera, target[0]!, target[1]!, target[2]!);
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
    setSceneCamera(world, scene, camera);
    const materials = world.resource(Materials);
    const geometry: ShapeGeometryCache = new Map();
    const presentation = scenePresentation(world, scene, geometry);
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
            placeMarker(world, camera, feedback.marker, p, 8);
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
    const shared = new Map<
        object,
        { mesh: MeshHandle; edges: ReturnType<typeof registerGizmo>; users: number }
    >();
    const meshKeys = new Map<MeshHandle, object>();
    const labels: number[] = [];
    const drivenSet = new Set<number>();
    function updateShapes(world: World): void {
        let visible: Set<number> | undefined;
        if (scene === "largeWorld") {
            const pose = world.get(camera, Transform)!;
            const forward = qRotate(...pose.rotation, 0, 0, -1);
            visible = drawSet(
                world,
                drawingBounds(
                    pose.translation,
                    forward,
                    typeof window === "undefined" ? 1 : window.innerWidth / window.innerHeight,
                    world.storage(Camera).fov.get(camera),
                ),
            );
        }
        for (const [ref, mesh] of views) {
            const eid = world.resolve(ref);
            if (!eid || !world.has(eid, Shape) || (visible && !visible.has(eid))) {
                if (eid) {
                    world.remove(eid, MeshInstance);
                    world.remove(eid, MeshMaterial);
                    world.remove(eid, Gizmo);
                }
                const key = meshKeys.get(mesh)!;
                const asset = shared.get(key)!;
                if (--asset.users === 0) {
                    meshes.remove(mesh);
                    meshKeys.delete(mesh);
                    shared.delete(key);
                    for (const [geometryKey, entry] of geometry)
                        if (entry.mesh === key) geometry.delete(geometryKey);
                }
                views.delete(ref);
            }
        }
        const sample = world.resource(EventSample);
        const driven = drivenBodies(world, drivenSet);
        for (const eid of visible ?? world.query([Shape])) {
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
            const role = shapeRole(world, eid, driven);
            if (views.has(ref)) {
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
            const shape = initial.get(ref) ?? shapePresentation(world, eid, geometry, driven);
            initial.delete(ref);
            let asset = shared.get(shape.mesh);
            if (!asset) {
                const edges = new GizmoAsset();
                for (const [a, b] of shape.edges) drawLine(edges, a, b, 0x000000);
                asset = {
                    mesh: meshes.add(shape.mesh),
                    edges: registerGizmo(world, edges),
                    users: 0,
                };
                meshKeys.set(asset.mesh, shape.mesh);
                shared.set(shape.mesh, asset);
            }
            const mesh = asset.mesh;
            asset.users++;
            views.set(ref, mesh);
            world.add(eid, Gizmo, {
                handle: asset.edges,
                width: 1,
                opacity: 0.45,
                depthBias: -0.0001,
            });
            world.add(eid, MeshInstance, { mesh });
            world.add(eid, MeshMaterial, materialFor(shape.role, eid));
        }
        drawSensorHitViews(world, camera, (a, b, color, options) =>
            drawLine(world, a, b, color, options),
        );
        const marks =
            sample.scene === "persistentContact"
                ? sample.contactMarks
                : sample.hits.map((hit) => ({
                      point: [hit.point.x, hit.point.y, hit.point.z] as [number, number, number],
                      end: [
                          hit.point.x - hit.approachSpeed * hit.normal.x,
                          hit.point.y - hit.approachSpeed * hit.normal.y,
                          hit.point.z - hit.approachSpeed * hit.normal.z,
                      ] as [number, number, number],
                      label: `${hit.approachSpeed.toFixed(1)}, ${hit.userMaterialIdA}`,
                  }));
        while (labels.length > marks.length) world.destroy(labels.pop()!);
        for (let i = 0; i < marks.length; i++) {
            const mark = marks[i]!;
            const point = mark.point;
            if (!labels[i]) {
                const label = world.create();
                labels[i] = label;
                world.add(label, Transform, {
                    translation: point,
                });
                world.add(label, Text, {
                    content: internText(world, mark.label),
                    fontSize: 1,
                    depthBias: -1,
                    billboard: Billboard.Screen,
                    color: 0xeee9df,
                });
                world.add(label, Anchor, { offset: [...Anchor.BOTTOM_LEFT] });
            }
            const label = labels[i]!;
            world.storage(Transform).translation.set(label, ...point);
            world.storage(Text).content.set(label, internText(world, mark.label));
            drawEventMark(mark, (a, b, color, options) => drawLine(world, a, b, color, options));
        }
    }
    updateShapes(world);
    addSystem({
        name: "shape-views",
        group: "simulation",
        after: OrbitPlugin.systems,
        update: updateShapes,
    });
    addSystem({
        name: "joint-anchors",
        group: "simulation",
        after: OrbitPlugin.systems,
        update(world) {
            drawJointViews(world, scene, camera, (a, b, color, options) =>
                drawLine(world, a, b, color, options),
            );
        },
    });
    if (scene === "joint") {
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
        for (const asset of shared.values()) meshes.remove(asset.mesh);
        meshes.remove(markerMesh);
        if (groundMesh !== undefined) meshes.remove(groundMesh);
        for (const material of stateMaterials.values()) materials.remove(material);
        materials.remove(markerMaterial);
        Object.assign(feedback, GrabFeedback.create());
    };
}
