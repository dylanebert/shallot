import {
    AmbientLight,
    Body,
    BodyType,
    Camera,
    Character,
    CharacterPlugin,
    component,
    cube,
    DirectionalLight,
    InputPlugin,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    MeshPlugin,
    mountOverlay,
    Player,
    PlayerPlugin,
    type Plugin,
    pointerLockRefusal,
    pointerLockStatus,
    type Resource,
    registerMesh,
    ShapeKind,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    type System,
    setKinematic,
    Transform,
    type World,
} from "@dylanebert/shallot";

type Vec4 = readonly [number, number, number, number];
const GROUND_COLOR = [0.15, 0.19, 0.2, 1] as const;
const STEP_COLOR = [0.28, 0.34, 0.34, 1] as const;
const LIFT_COLOR = [0.83, 0.53, 0.24, 1] as const;
const TOWER_COLOR = [0.34, 0.4, 0.39, 1] as const;
const PERCH_COLOR = [0.39, 0.45, 0.43, 1] as const;

function block(
    world: World,
    at: Vec4,
    size: Vec4,
    rgba: Vec4,
    type: BodyType = BodyType.Static,
): number {
    const eid = world.create();
    world.add(eid, Body, { type, position: at, halfExtents: size });
    const name = `block-${eid}`;
    registerMesh(world, { name, ...cube([size[0], size[1], size[2]]) });
    world.add(eid, MeshInstance, { mesh: world.resource(Meshes).id(name)! });
    world.add(eid, MeshMaterial, {
        material: world.resource(Materials).add(StandardMaterial({ baseColor: rgba })),
    });
    return eid;
}

export const Route: Resource<{ entities: ReturnType<typeof route> | null }> = {
    create: () => ({ entities: null }),
};

export function route(world: World) {
    const ambient = world.create();
    world.add(ambient, AmbientLight, { color: 0xd6dfdf, intensity: 0.74 });
    const sun = world.create();
    world.add(sun, DirectionalLight, {
        direction: [-0.45, -1, -0.6, 0],
        color: 0xffe8c7,
        intensity: 1.1,
    });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const eye = world.create();
    world.add(eye, Camera);
    world.add(eye, StandardRenderer);
    world.add(eye, Transform, { translation: [0, 2.7, 12, 0] });
    const player = world.create();
    world.add(player, Body, {
        position: [0, 2, 12, 0],
        shape: ShapeKind.Capsule,
        halfExtents: [0, 0.6, 0, 0.3],
        type: BodyType.Kinematic,
    });
    // The route keeps its launch and fall rhythm; pogo placement floats two radii above contact.
    world.add(player, Character);
    world.add(player, Player, { camera: eye, jumpSpeed: 7, gravity: 30 });
    block(world, [0, 0, -5, 0], [16, 0.5, 26, 0], GROUND_COLOR);
    const steps = [
        [
            [0, 0.75, 3, 0],
            [3, 0.25, 1.5, 0],
        ],
        [
            [0, 1.25, 0, 0],
            [3, 0.25, 1.5, 0],
        ],
        [
            [0, 1.75, -3, 0],
            [3, 0.25, 1.5, 0],
        ],
    ] as const satisfies readonly (readonly [Vec4, Vec4])[];
    for (const [at, size] of steps) block(world, at, size, STEP_COLOR);
    const lift = block(world, [0, 1.75, -6.5, 0], [3, 0.25, 2, 0], LIFT_COLOR, BodyType.Kinematic);
    world.add(lift, Lift);
    const tower = [
        [[0.7, 2.5, -10, 0], [2.2, 0.5, 1.2, 0], TOWER_COLOR],
        [[-0.7, 3.5, -12.3, 0], [1.8, 0.5, 1.2, 0], TOWER_COLOR],
        [[0.7, 4.5, -14.5, 0], [2.2, 0.5, 1.5, 0], PERCH_COLOR],
    ] as const satisfies readonly (readonly [Vec4, Vec4, Vec4])[];
    for (const [at, size, rgba] of tower) block(world, at, size, rgba);
    return { player, eye, lift };
}

// The route owns the lift's size and starting height. This role only gives the small trajectory system a
// declarative target; the lift is the sole moving object in the recipe.
export const Lift = component("Lift", {});

const TRAVEL = 1.5;
const RATE = 0.65;
const RECIPE_STATE: Resource<DemoBag> = { create: createBag };
type DemoBag = {
    // one slot per lift: its body eid, and its authored base at `slot * 3`. Two held arrays rather than a
    // Map, so the per-tick walk indexes instead of iterating and the base reads stay unboxed doubles.
    liftEids: number[];
    liftBases: number[];
    liftCount: number;
    panel: HTMLDivElement | null;
    look: HTMLDivElement | null;
};
function stateBag(world: World): DemoBag {
    return world.resource(RECIPE_STATE);
}

// The bag's creation, apart from the per-frame lookup: its dispose closure would otherwise make every
// lookup allocate a context.
function createBag(world: World): DemoBag {
    const bag: DemoBag = {
        liftEids: [],
        liftBases: [],
        liftCount: 0,
        panel: null,
        look: null,
    };
    world.onDispose(() => {
        // the slot arrays keep their capacity; the count is what empties them
        bag.liftCount = 0;
        bag.panel = null;
        bag.look = null;
    });
    return bag;
}

// The lift's pose and velocity registers, written in place each tick; setKinematic copies them.
const liftPos: [number, number, number] = [0, 0, 0];
const LIFT_QUAT = [0, 0, 0, 1] as const;
const liftVel: [number, number, number] = [0, 0, 0];

const lift: System = {
    name: "lift",
    group: "fixed",
    before: CharacterPlugin.systems,
    // Every lift shares one trajectory, so the phase, the rise and the velocity are the tick's, not each
    // lift's: they are computed once here and the slot walk only adds each lift's base to them.
    update(world: World): void {
        const bag = stateBag(world);
        const phase = 2 * (world.time.elapsed * RATE);
        const rise = 0.5 * TRAVEL * (1 - Math.cos(phase));
        liftVel[1] = RATE * TRAVEL * Math.sin(phase);
        for (let slot = 0; slot < bag.liftCount; slot++) {
            const base = slot * 3;
            liftPos[0] = bag.liftBases[base];
            liftPos[1] = bag.liftBases[base + 1] + rise;
            liftPos[2] = bag.liftBases[base + 2];
            setKinematic(world, bag.liftEids[slot], liftPos, LIFT_QUAT, false, liftVel);
        }
    },
};

function mountControls(world: World): void {
    if (typeof document === "undefined") return;
    const bag = stateBag(world);
    if (bag.panel) return;
    const overlay = mountOverlay(document.querySelector("canvas"), world);
    const panel = document.createElement("div");
    panel.dataset.recipeControls = "";
    panel.style.cssText =
        "position:absolute;top:20px;left:20px;pointer-events:none;padding:10px 12px;" +
        "display:grid;row-gap:6px;column-gap:16px;border:1px solid rgba(255,255,255,0.12);" +
        "border-radius:6px;background:rgba(14,17,20,0.72);color:#ffffff;" +
        "font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace";
    for (const [control, action] of [
        ["WASD", "Move"],
        ["MOUSE", "Look"],
        ["SPACE", "Jump"],
    ] as const) {
        const row = document.createElement("div");
        row.dataset.controlRow = "";
        row.style.cssText =
            "display:grid;grid-template-columns:max-content max-content;column-gap:16px";
        for (const text of [control, action]) {
            const cell = document.createElement("span");
            cell.textContent = text;
            cell.style.color = "#ffffff";
            row.append(cell);
        }
        panel.append(row);
        if (control === "MOUSE") bag.look = row;
    }
    overlay.append(panel);
    bag.panel = panel;
}

// Pointer look states itself on the control it governs, never as a sentence: the MOUSE row is dim until the
// pointer locks and brightens when it does, so the affordance and its outcome are one mark. A refusal reads
// as a struck row, with the browser's reason kept on the title so the cause stays recoverable without copy.
const controls: System = {
    name: "first-person-controls",
    group: "draw",
    update(world) {
        mountControls(world);
        const bag = stateBag(world);
        if (!bag.panel || !bag.look) return;
        const status = pointerLockStatus(world);
        const look =
            status === "locked" ? "locked" : status === "unlocked" ? "idle" : "unavailable";
        if (bag.look.dataset.pointerLook === look) return;
        bag.look.dataset.pointerLook = look;
        bag.look.style.opacity = look === "locked" ? "1" : "0.45";
        bag.look.style.textDecoration = look === "unavailable" ? "line-through" : "none";
        bag.look.title = look === "unavailable" ? (pointerLockRefusal(world) ?? "") : "";
    },
};

export const Demo = {
    name: "Demo",
    components: [Lift],
    dependencies: [PlayerPlugin, CharacterPlugin, InputPlugin, StandardPhysicsPlugin, MeshPlugin],
    initialize(world: World) {
        const state = world.resource(Route);
        state.entities ??= route(world);
    },
    warm(world: World) {
        const bag = stateBag(world);
        bag.liftCount = 0;
        for (const eid of world.query([Lift, Body])) {
            const base = bag.liftCount * 3;
            bag.liftEids[bag.liftCount] = eid;
            bag.liftBases[base] = world.storage(Body).position.x.get(eid);
            bag.liftBases[base + 1] = world.storage(Body).position.y.get(eid);
            bag.liftBases[base + 2] = world.storage(Body).position.z.get(eid);
            bag.liftCount++;
        }
    },
    systems: [lift, controls],
} satisfies Plugin;
