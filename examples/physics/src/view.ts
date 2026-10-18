import {
    Body,
    BodyType,
    Camera,
    cube,
    DirectionalLight,
    GlobalAmbientLight,
    Hulls,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    NotShadowCaster,
    Shape,
    ShapeKind,
    StandardMaterial,
    StandardRenderer,
    sphere,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { Grab } from "./grab";
import { type Scene, scenes } from "./scenes";

export const GrabFeedback = { create: () => ({ marker: 0 }) };

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
    const ground = materials.add(
        StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1], perceptualRoughness: 0.82 }),
    );
    const subject = materials.add(
        StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 0.38 }),
    );
    const markerMesh = world.resource(Meshes).add({ name: "Grab point", ...sphere(0.3, 16, 12) });
    const markerMaterial = materials.add(
        StandardMaterial({ baseColor: [0.855, 0.815, 0.738, 1], unlit: true }),
    );
    const feedback = world.resource(GrabFeedback);
    world.addSystem({
        name: "grab-feedback",
        group: "fixed",
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
            world.storage(Transform).translation.set(feedback.marker, ...grab.point);
        },
    });
    const shapes = world.storage(Shape);
    const types = world.storage(Body).type;
    const meshes = world.resource(Meshes);
    for (const eid of world.query([Shape])) {
        const hulls = world.resource(Hulls);
        const hull = hulls.get(hulls.name(shapes.geometry.get(eid))!);
        if (shapes.kind.get(eid) !== ShapeKind.Hull || !hull?.halfExtents || shapes.body.get(eid))
            throw new Error("The physics viewer currently draws body-local box hulls only");
        const size = hull.halfExtents.map(
            (extent, axis) =>
                2 * extent * [shapes.scale.x, shapes.scale.y, shapes.scale.z][axis]!.get(eid),
        ) as [number, number, number];
        const mesh = meshes.add({ name: `Shape ${eid}`, ...cube(size) });
        world.add(eid, MeshInstance, { mesh });
        // Colour encodes affordance and emphasis: static bodies are carbon, grabbable dynamics amber;
        // cyan and brand-orange are reserved for distinct scene roles, never creation order.
        world.add(eid, MeshMaterial, types.get(eid) === BodyType.Static ? ground : subject);
    }
}
