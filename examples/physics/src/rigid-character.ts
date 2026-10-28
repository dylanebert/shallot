import { Body, GlobalTransform, Shape, Transform, type World } from "@dylanebert/shallot";
import {
    castShape,
    getMassData,
    setLinearVelocity,
    setMassData,
    setTransform,
} from "@dylanebert/shallot/standard/physics";
import { add, length, mul, sub, type Vec, v } from "./mover";

const f = Math.fround;
export const SRC = f(0.0254);
export const BODY_RADIUS = f(16 * SRC);
export const TOTAL_HEIGHT = f(72 * SRC);
export const FEET_HEIGHT = f(TOTAL_HEIGHT * 0.5);
const WALK_SPEED = f(230 * SRC),
    RUN_SPEED = f(350 * SRC),
    JUMP_SPEED = f(300 * SRC);
const STEP_HEIGHT = f(18 * SRC),
    SKIN = f(f(0.095) * SRC);
const BRAKE_POWER = f(0.2),
    SURFACE_FRICTION = f(0.6);
export const rigidCharacter = () => ({
    body: 0,
    feet: 0,
    capsule: 0,
    position: v(7.5, 2, 9),
    velocity: v(),
    onGround: false,
    sprint: false,
    jumpCooldown: 0,
    groundNormal: v(0, 1, 0),
    groundVelocity: v(),
    didStep: false,
    stepPosition: v(),
    wishVelocity: v(),
    pendingPosition: false,
    pendingVelocity: false,
    initialized: false,
});
export type RigidCharacter = ReturnType<typeof rigidCharacter>;
const standable = (normal: Vec) => normal.y >= f(Math.cos(f(f(45 * f(Math.PI)) / 180)));
function readVelocity(world: World, s: RigidCharacter): Vec {
    const g = world.storage(GlobalTransform).linearVelocity;
    return v(g.x.get(s.body), g.y.get(s.body), g.z.get(s.body));
}
function readPosition(world: World, s: RigidCharacter): Vec {
    const g = world.storage(GlobalTransform).translation;
    return v(g.x.get(s.body), g.y.get(s.body), g.z.get(s.body));
}
function massCenter(world: World, s: RigidCharacter, wishSpeed: number) {
    const mass = getMassData(world, s.body);
    if (!mass) return;
    const half = f(TOTAL_HEIGHT * 0.5);
    mass.center = s.onGround ? v(0, f(Math.min(Math.max(wishSpeed, 0), half) - half), 0) : v();
    setMassData(world, s.body, mass);
}
function trace(
    world: World,
    s: RigidCharacter,
    from: Vec,
    to: Vec,
    radiusScale = 1,
    heightScale = 1,
) {
    const tr = {
        endPosition: to,
        normal: v(0, 1, 0),
        fraction: 1,
        hit: false,
        startedSolid: false,
    };
    const translation = sub(to, from);
    if (length(translation) < f(1e-6)) return tr;
    const halfW = f(f(BODY_RADIUS * 0.5) * radiusScale),
        halfH = f(f(TOTAL_HEIGHT * heightScale) * 0.5);
    const centerY = f(f(TOTAL_HEIGHT * heightScale) * 0.5);
    const points = Array.from({ length: 8 }, (_, i) =>
        v(i & 1 ? halfW : -halfW, f(centerY + (i & 2 ? halfH : -halfH)), i & 4 ? halfW : -halfW),
    );
    castShape(world, from, { points, count: 8, radius: 0 }, translation, (hit) => {
        if (hit.shape === s.feet || hit.shape === s.capsule) return -1;
        if (hit.fraction === 0) {
            tr.startedSolid = true;
            return -1;
        }
        if (hit.fraction < tr.fraction) {
            tr.fraction = hit.fraction;
            tr.normal = { ...hit.normal };
            tr.hit = true;
            tr.endPosition = add(from, mul(hit.fraction, translation));
        }
        return tr.fraction;
    });
    return tr;
}
function ground(s: RigidCharacter, onGround: boolean, normal = v(0, 1, 0)) {
    s.onGround = onGround;
    s.groundNormal = normal;
    if (!onGround) s.groundVelocity = v();
}
// samples/sample_character.cpp RigidbodyCharacter: the character's choices stay in the game.
export function prepareRigid(
    world: World,
    s: RigidCharacter,
    forward: Vec,
    right: Vec,
    keys: Record<"W" | "A" | "S" | "D" | "Space" | "Shift", number>,
) {
    if (s.pendingPosition) {
        setTransform(world, s.body, [s.position.x, s.position.y, s.position.z], [0, 0, 0, 1]);
        s.pendingPosition = false;
    }
    if (s.pendingVelocity) {
        setLinearVelocity(world, s.body, s.velocity);
        s.pendingVelocity = false;
    }
    s.velocity = readVelocity(world, s);
    if (keys.Space && s.onGround && s.jumpCooldown <= 0) {
        s.velocity.y = JUMP_SPEED;
        setLinearVelocity(world, s.body, s.velocity);
        s.onGround = false;
        s.jumpCooldown = f(0.2);
    }
    s.sprint = s.onGround && !!keys.Shift;
    const dt = f(1 / 60);
    if (s.jumpCooldown > 0) s.jumpCooldown = f(s.jumpCooldown - dt);
    forward = v(forward.x, 0, forward.z);
    const forwardLength = length(forward);
    if (forwardLength > f(0.001)) forward = mul(f(1 / forwardLength), forward);
    const speed = s.sprint ? RUN_SPEED : WALK_SPEED;
    let wish = add(
        mul(f(speed * (keys.W - keys.S)), forward),
        mul(f(speed * (keys.D - keys.A)), right),
    );
    const wishSpeed = length(wish);
    if (wishSpeed > speed) wish = mul(f(speed / wishSpeed), wish);
    s.wishVelocity = wish;
    const wishLen = length(wish),
        velLen = length(s.velocity);
    const wantsBrakes = s.onGround && (wishLen < f(5 * SRC) || wishLen < f(velLen * f(0.9)));
    world
        .storage(Shape)
        .friction.set(s.feet, wantsBrakes ? f(1 + f(f(100 * BRAKE_POWER) * SURFACE_FRICTION)) : 0);
    massCenter(world, s, wishLen);
    const groundSpeed = length(s.groundVelocity);
    world
        .storage(Body)
        .gravityScale.set(
            s.body,
            !s.onGround || velLen > SRC || groundSpeed > SRC ? f(15 / 10) : 0,
        );
    world
        .storage(Body)
        .linearDamping.set(
            s.body,
            s.onGround && wishLen < SRC && groundSpeed < SRC ? f(10 * BRAKE_POWER) : f(0.1),
        );
}
export function stepRigid(world: World, s: RigidCharacter) {
    if (!s.initialized) {
        massCenter(world, s, 0);
        s.initialized = true;
    }
    s.position = readPosition(world, s);
    let vel = readVelocity(world, s);
    const wish = v(s.wishVelocity.x, 0, s.wishVelocity.z),
        wishLen = length(wish);
    if (wishLen >= f(0.001)) {
        const savedY = vel.y;
        let velocity = sub(vel, s.groundVelocity);
        const maxSpeed = Math.max(wishLen, length(velocity));
        const amount = s.onGround ? f(f(0.25) + f(SURFACE_FRICTION * 10)) : f(0.05);
        let addition = mul(amount, wish);
        const maxAdd = s.onGround ? f(wishLen * amount) : wishLen;
        const addLen = length(addition);
        if (addLen > maxAdd && addLen > 0) addition = mul(f(maxAdd / addLen), addition);
        velocity = add(velocity, addition);
        const newSpeed = length(velocity);
        if (newSpeed > maxSpeed && newSpeed > 0) velocity = mul(f(maxSpeed / newSpeed), velocity);
        vel = add(velocity, s.groundVelocity);
        if (s.onGround) vel.y = savedY;
        setLinearVelocity(world, s.body, vel);
    }
    s.didStep = false;
    if (!s.onGround) return;
    const horizontal = v(vel.x, 0, vel.z),
        speed = length(horizontal);
    if (speed < f(0.01)) return;
    const direction = v(f(horizontal.x / speed), 0, f(horizontal.z / speed));
    const distance = f(f(speed * f(1 / 60)) + BODY_RADIUS);
    const from = sub(s.position, mul(SKIN, direction)),
        to = add(s.position, mul(distance, direction));
    let radius = 1,
        ahead = trace(world, s, from, to, radius);
    while (ahead.startedSolid) {
        radius = f(radius - f(0.1));
        if (radius < f(0.6)) return;
        ahead = trace(world, s, from, to, radius);
    }
    if (!ahead.hit) return;
    const upFrom = ahead.endPosition,
        upTo = add(upFrom, v(0, STEP_HEIGHT, 0));
    const up = trace(world, s, upFrom, upTo, radius);
    if (up.startedSolid) return;
    const top = up.hit ? up.endPosition : upTo;
    if (f(top.y - upFrom.y) < f(0.005)) return;
    const acrossDistance = f(f(distance * f(1 - ahead.fraction)) + f(BODY_RADIUS * 0.5));
    const acrossTo = add(top, mul(acrossDistance, direction)),
        across = trace(world, s, top, acrossTo, radius);
    if (across.startedSolid) return;
    const acrossPos = across.hit ? across.endPosition : acrossTo;
    const down = trace(world, s, acrossPos, sub(acrossPos, v(0, STEP_HEIGHT, 0)), radius);
    if (!down.hit || !standable(down.normal) || f(down.endPosition.y - s.position.y) < f(0.01))
        return;
    s.stepPosition = add(down.endPosition, v(0, f(0.01), 0));
    setTransform(
        world,
        s.body,
        [s.stepPosition.x, s.stepPosition.y, s.stepPosition.z],
        [0, 0, 0, 1],
    );
    setLinearVelocity(world, s.body, v(f(vel.x * f(0.9)), 0, f(vel.z * f(0.9))));
    s.didStep = true;
}
export function lateRigid(world: World, s: RigidCharacter) {
    s.position = readPosition(world, s);
    s.velocity = readVelocity(world, s);
    if (s.didStep) {
        s.position = { ...s.stepPosition };
        s.pendingPosition = true;
        s.didStep = false;
    }
    if (s.onGround) {
        const from = add(s.position, v(0, f(0.05), 0)),
            to = sub(s.position, v(0, STEP_HEIGHT, 0));
        let radius = 1,
            tr = trace(world, s, from, to, radius, 0.5);
        while (tr.startedSolid) {
            radius = f(radius - f(0.1));
            if (radius < f(0.7)) break;
            tr = trace(world, s, from, to, radius, 0.5);
        }
        if (radius >= f(0.7) && tr.hit) {
            const target = add(tr.endPosition, v(0, f(0.01), 0));
            if (f(target.y - s.position.y) > f(0.01)) {
                s.velocity.y = 0;
                s.pendingVelocity = true;
            }
            s.position = target;
            s.pendingPosition = true;
        }
    }
    if (s.pendingPosition)
        world.storage(Transform).translation.set(s.body, s.position.x, s.position.y, s.position.z);
    const feet = sub(s.position, v(0, f(TOTAL_HEIGHT * 0.5), 0));
    const from = add(feet, v(0, f(4 * SRC), 0)),
        to = sub(feet, v(0, f(2 * SRC), 0));
    let radius = 1,
        tr = trace(world, s, from, to, radius, 0.5);
    while (tr.startedSolid || (tr.hit && !standable(tr.normal))) {
        radius = f(radius - f(0.1));
        if (radius < f(0.7)) {
            ground(s, false);
            return;
        }
        tr = trace(world, s, from, to, radius, 0.5);
    }
    const onGround = !tr.startedSolid && tr.hit && standable(tr.normal) && s.jumpCooldown <= 0;
    ground(s, onGround, onGround ? tr.normal : v(0, 1, 0));
}
