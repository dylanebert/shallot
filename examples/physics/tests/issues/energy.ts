import { GlobalTransform, type World } from "@dylanebert/shallot";
import {
    getAngularVelocity,
    getBodyTransform,
    getMassData,
    PhysicsWorldDefinition,
} from "@dylanebert/shallot/standard/physics";

const f = Math.fround;
type V = { x: number; y: number; z: number };
const dot = (a: V, b: V) => f(f(f(a.x * b.x) + f(a.y * b.y)) + f(a.z * b.z));
const cross = (a: V, b: V): V => ({
    x: f(f(a.y * b.z) - f(a.z * b.y)),
    y: f(f(a.z * b.x) - f(a.x * b.z)),
    z: f(f(a.x * b.y) - f(a.y * b.x)),
});

/** Mechanical energy in the body's mass frame, as measured by Box3D's sample. */
export function measureEnergy(world: World, body: number) {
    const mass = getMassData(world, body)!;
    const pose = getBodyTransform(world, body)!;
    const omega = getAngularVelocity(world, body)!;
    const q = pose.q;
    const t1 = cross(q.v, omega);
    const t2 = {
        x: f(t1.x - f(q.s * omega.x)),
        y: f(t1.y - f(q.s * omega.y)),
        z: f(t1.z - f(q.s * omega.z)),
    };
    const t3 = cross(q.v, t2);
    const w = {
        x: f(omega.x + f(2 * t3.x)),
        y: f(omega.y + f(2 * t3.y)),
        z: f(omega.z + f(2 * t3.z)),
    };
    const i = mass.inertia;
    const iw = {
        x: f(f(f(i.cx.x * w.x) + f(i.cy.x * w.y)) + f(i.cz.x * w.z)),
        y: f(f(f(i.cx.y * w.x) + f(i.cy.y * w.y)) + f(i.cz.y * w.z)),
        z: f(f(f(i.cx.z * w.x) + f(i.cy.z * w.y)) + f(i.cz.z * w.z)),
    };
    const velocity = world.storage(GlobalTransform).linearVelocity;
    const v = { x: velocity.x.get(body), y: velocity.y.get(body), z: velocity.z.get(body) };
    const gravity = world.resource(PhysicsWorldDefinition).gravity;
    const linear = f(f(0.5 * mass.mass) * dot(v, v));
    const angular = f(0.5 * dot(w, iw));
    const potential = f(-f(mass.mass * dot(gravity, pose.p)));
    return { linear, angular, potential, total: f(f(linear + angular) + potential) };
}
