import {
    Transform,
    WorldTransform,
    ChildOf,
    Camera,
    Viewport,
    PhysicsPlugin,
    Body,
    Move,
    BallJoint,
    Character,
    Part,
    Shape,
    Shadows,
    Tonemap,
    AmbientLight,
    DirectionalLight,
    RenderPlugin,
    Player,
    PlayerPlugin,
    Inputs,
    ActiveCamera,
    raycast,
    not,
} from "@dylanebert/shallot";
import { Outline, OutlinePlugin } from "@dylanebert/shallot/extras";
import type { Plugin, State, System, Ray } from "@dylanebert/shallot";
import { BenchConfig } from "../config";

const GRAB_MAX_DISTANCE = 5;

let crosshairEl: HTMLElement | null = null;

function setCrosshair(mode: "default" | "hover" | "grab") {
    if (!crosshairEl) return;
    const circles = crosshairEl.querySelectorAll("circle");
    const ring = circles[0];
    const dot = circles[1];
    const lines = Array.from(crosshairEl.querySelectorAll("line"));
    if (!ring || !dot) return;
    const showLines = mode === "default";
    ring.setAttribute("display", mode === "default" ? "none" : "");
    dot.setAttribute("display", mode === "grab" ? "" : "none");
    for (const l of lines) l.setAttribute("display", showLines ? "" : "none");
}

export function createCrosshair(container: HTMLElement) {
    const el = document.createElement("div");
    el.style.cssText =
        "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none;width:24px;height:24px;";
    const svg = `<svg viewBox="0 0 24 24" width="24" height="24" stroke="white" stroke-width="2" fill="none" opacity="0.8">
        <line x1="12" y1="4" x2="12" y2="10"/><line x1="12" y1="14" x2="12" y2="20"/>
        <line x1="4" y1="12" x2="10" y2="12"/><line x1="14" y1="12" x2="20" y2="12"/>
        <circle cx="12" cy="12" r="6" display="none"/>
        <circle cx="12" cy="12" r="2" fill="white" stroke="none" display="none"/>
    </svg>`;
    el.innerHTML = svg;
    crosshairEl = el;
    container.appendChild(el);
    return () => {
        el.remove();
        crosshairEl = null;
    };
}

function addStaticBox(
    state: State,
    x: number,
    y: number,
    z: number,
    sx: number,
    sy: number,
    sz: number,
    color: number,
): number {
    const eid = state.addEntity();
    state.addComponent(eid, Transform);
    state.addComponent(eid, Part);
    state.addComponent(eid, Body);
    Transform.posX[eid] = x;
    Transform.posY[eid] = y;
    Transform.posZ[eid] = z;
    Part.shape[eid] = Shape.Box;
    Part.sizeX[eid] = sx;
    Part.sizeY[eid] = sy;
    Part.sizeZ[eid] = sz;
    Part.color[eid] = color;
    Body.mass[eid] = 0;
    Body.friction[eid] = 0.5;
    return eid;
}

function addDynamicBox(
    state: State,
    x: number,
    y: number,
    z: number,
    size: number,
    color: number,
): number {
    const eid = state.addEntity();
    state.addComponent(eid, Transform);
    state.addComponent(eid, Part);
    state.addComponent(eid, Body);
    Transform.posX[eid] = x;
    Transform.posY[eid] = y;
    Transform.posZ[eid] = z;
    Part.shape[eid] = Shape.Box;
    Part.sizeX[eid] = size;
    Part.sizeY[eid] = size;
    Part.sizeZ[eid] = size;
    Part.color[eid] = color;
    Body.mass[eid] = 1;
    Body.friction[eid] = 0.5;
    return eid;
}

let hPlatformEid = -1;
let vPlatformEid = -1;
let kLeverEid = -1;
let anchorEid = -1;
let grabJointEid = -1;
let dragTarget = -1;
let dragRayDistance = 0;
let dragLocalAnchor: [number, number, number] = [0, 0, 0];
let wasLeftDown = false;
let hoveredEid = -1;

const ray: Ray = { origin: { x: 0, y: 0, z: 0 }, direction: { x: 0, y: 0, z: 0 } };

function updateCenterRay(camEid: number) {
    const d = WorldTransform.data;
    const off = camEid * 16;
    let dx = -d[off + 8];
    let dy = -d[off + 9];
    let dz = -d[off + 10];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    dx /= len;
    dy /= len;
    dz /= len;
    const near = Camera.near[camEid];
    ray.origin.x = d[off + 12] + dx * near;
    ray.origin.y = d[off + 13] + dy * near;
    ray.origin.z = d[off + 14] + dz * near;
    ray.direction.x = dx;
    ray.direction.y = dy;
    ray.direction.z = dz;
}

function worldToLocal(
    eid: number,
    point: { x: number; y: number; z: number },
): [number, number, number] {
    const px = point.x - Transform.posX[eid];
    const py = point.y - Transform.posY[eid];
    const pz = point.z - Transform.posZ[eid];
    const qx = -Transform.quatX[eid];
    const qy = -Transform.quatY[eid];
    const qz = -Transform.quatZ[eid];
    const qw = Transform.quatW[eid];
    const tx = 2 * (qy * pz - qz * py);
    const ty = 2 * (qz * px - qx * pz);
    const tz = 2 * (qx * py - qy * px);
    return [
        px + qw * tx + qy * tz - qz * ty,
        py + qw * ty + qz * tx - qx * tz,
        pz + qw * tz + qx * ty - qy * tx,
    ];
}

const GrabSystem: System = {
    setup(state) {
        state.setResource(Outline, {
            getEntities: () => (hoveredEid >= 0 ? [hoveredEid] : []),
            color: 0xffffff,
            thickness: 2,
        });
    },
    update(state) {
        const input = Inputs.from(state);
        const cam = ActiveCamera.from(state);
        if (!input || !cam || cam.eid < 0) return;

        const mouse = input.mouse;
        const leftDown = mouse.left;
        const leftPressed = leftDown && !wasLeftDown;
        const leftReleased = !leftDown && wasLeftDown;
        wasLeftDown = leftDown;

        updateCenterRay(cam.eid);

        if (leftPressed && dragTarget < 0) {
            const hit = raycast(state.query([Part, Transform, Body, not(Character)]), ray);
            if (hit && Body.mass[hit.eid] > 0 && hit.distance <= GRAB_MAX_DISTANCE) {
                dragTarget = hit.eid;
                dragLocalAnchor = worldToLocal(hit.eid, hit.point);
                dragRayDistance = Math.max(
                    (hit.point.x - ray.origin.x) * ray.direction.x +
                        (hit.point.y - ray.origin.y) * ray.direction.y +
                        (hit.point.z - ray.origin.z) * ray.direction.z,
                    0.1,
                );
                Transform.posX[anchorEid] = hit.point.x;
                Transform.posY[anchorEid] = hit.point.y;
                Transform.posZ[anchorEid] = hit.point.z;

                grabJointEid = state.addEntity();
                state.addComponent(grabJointEid, BallJoint);
                BallJoint.bodyA[grabJointEid] = anchorEid;
                BallJoint.bodyB[grabJointEid] = dragTarget;
                BallJoint.anchorBX[grabJointEid] = dragLocalAnchor[0];
                BallJoint.anchorBY[grabJointEid] = dragLocalAnchor[1];
                BallJoint.anchorBZ[grabJointEid] = dragLocalAnchor[2];
                BallJoint.stiffness[grabJointEid] = 5000;
                setCrosshair("grab");
            }
        }

        if (dragTarget >= 0 && leftDown) {
            Transform.posX[anchorEid] = ray.origin.x + ray.direction.x * dragRayDistance;
            Transform.posY[anchorEid] = ray.origin.y + ray.direction.y * dragRayDistance;
            Transform.posZ[anchorEid] = ray.origin.z + ray.direction.z * dragRayDistance;
        }

        if (leftReleased && dragTarget >= 0) {
            dragTarget = -1;
            Transform.posY[anchorEid] = -1000;
            state.removeEntity(grabJointEid);
            grabJointEid = -1;
            setCrosshair("default");
        }

        if (dragTarget < 0) {
            const hit = raycast(state.query([Part, Transform, Body, not(Character)]), ray);
            hoveredEid =
                hit && Body.mass[hit.eid] > 0 && hit.distance <= GRAB_MAX_DISTANCE ? hit.eid : -1;
            setCrosshair(hoveredEid >= 0 ? "hover" : "default");
        }
    },
};

const PlatformSystem: System = {
    group: "fixed",
    update(state) {
        const t = state.time.elapsed;
        if (hPlatformEid >= 0) {
            Transform.posX[hPlatformEid] = -12 + Math.sin(t * 0.8) * 5;
            Transform.posY[hPlatformEid] = 1;
            Transform.posZ[hPlatformEid] = 12;
        }
        if (vPlatformEid >= 0) {
            Transform.posX[vPlatformEid] = -12;
            Transform.posY[vPlatformEid] = 2 + Math.sin(t * 0.6) * 1.5;
            Transform.posZ[vPlatformEid] = 6;
        }
        if (kLeverEid >= 0) {
            const ha = t * 0.25;
            Transform.posX[kLeverEid] = 8;
            Transform.posY[kLeverEid] = 1;
            Transform.posZ[kLeverEid] = -18;
            Transform.quatY[kLeverEid] = Math.sin(ha);
            Transform.quatW[kLeverEid] = Math.cos(ha);
        }
    },
};

export function buildPlayerPlugin(): Plugin {
    return {
        name: "PlayerScenario",
        dependencies: [RenderPlugin, PhysicsPlugin, PlayerPlugin, OutlinePlugin],
        systems: [PlatformSystem, GrabSystem],
        initialize(state: State) {
            const body = state.addEntity();
            state.addComponent(body, Transform);
            state.addComponent(body, Part);
            state.addComponent(body, Body);
            state.addComponent(body, Character);
            state.addComponent(body, Player);
            Transform.posX[body] = 0;
            Transform.posY[body] = 0.9;
            Transform.posZ[body] = 0;
            Part.shape[body] = Shape.Box;
            Part.sizeX[body] = 0.6;
            Part.sizeY[body] = 1.8;
            Part.sizeZ[body] = 0.6;
            Part.color[body] = 0x000000;
            Part.opacity[body] = 0;
            Body.mass[body] = 0;
            Body.friction[body] = 0.5;

            const cam = state.addEntity();
            state.addComponent(cam, Transform);
            state.addComponent(cam, Camera);
            state.addComponent(cam, Viewport);
            state.addComponent(cam, Tonemap);
            state.addComponent(cam, Shadows);
            state.addComponent(cam, BenchConfig);
            state.addRelation(cam, ChildOf, body);

            const ambient = state.addEntity();
            state.addComponent(ambient, Transform);
            state.addComponent(ambient, AmbientLight);

            const sun = state.addEntity();
            state.addComponent(sun, Transform);
            state.addComponent(sun, DirectionalLight);
            DirectionalLight.directionX[sun] = -0.5;
            DirectionalLight.directionY[sun] = -0.7;
            DirectionalLight.directionZ[sun] = -0.5;

            addStaticBox(state, 0, -0.5, 0, 50, 1, 50, 0x252220);

            addStaticBox(state, 0, 2, -25, 50, 5, 1, 0x3a3530);
            addStaticBox(state, 0, 2, 25, 50, 5, 1, 0x3a3530);
            addStaticBox(state, -25, 2, 0, 1, 5, 50, 0x3a3530);
            addStaticBox(state, 25, 2, 0, 1, 5, 50, 0x3a3530);

            addStaticBox(state, 5, 2, -5, 1, 4, 1, 0x504a44);
            addStaticBox(state, -5, 2, -5, 1, 4, 1, 0x504a44);
            addStaticBox(state, 5, 2, 5, 1, 4, 1, 0x504a44);
            addStaticBox(state, -5, 2, 5, 1, 4, 1, 0x504a44);

            addStaticBox(state, 10, 1, 0, 6, 2, 6, 0x453f3a);
            addStaticBox(state, 10, 3, -8, 4, 1, 4, 0x453f3a);

            addStaticBox(state, 10, 0.5, 5, 3, 0.2, 5, 0x4a4540);
            addStaticBox(state, 10, 1.0, 4, 3, 0.2, 4, 0x4a4540);
            addStaticBox(state, 10, 1.5, 3, 3, 0.2, 3, 0x4a4540);

            addStaticBox(state, -10, 1.5, -3, 0.5, 3, 8, 0x504a44);
            addStaticBox(state, -7, 1.5, -3, 0.5, 3, 8, 0x504a44);

            addStaticBox(state, 0, 1, -10, 8, 2, 0.5, 0x504a44);
            addStaticBox(state, 4, 1, -15, 0.5, 2, 10, 0x504a44);

            addDynamicBox(state, 3, 0.5, 3, 1, 0xd49560);
            addDynamicBox(state, -3, 0.5, 2, 0.8, 0xd49560);
            addDynamicBox(state, 1, 0.5, -3, 1.2, 0xd49560);
            addDynamicBox(state, -2, 0.5, -7, 0.6, 0xd49560);
            addDynamicBox(state, 2, 0.5, -12, 1, 0xd49560);

            addDynamicBox(state, 15, 0.5, 10, 1, 0xd49560);
            addDynamicBox(state, 15, 1.5, 10, 1, 0xd49560);
            addDynamicBox(state, 15, 2.5, 10, 1, 0xd49560);

            hPlatformEid = addStaticBox(state, -12, 1, 12, 6, 0.4, 3, 0x4078a0);
            state.addComponent(hPlatformEid, Move);
            addDynamicBox(state, -12, 1.7, 12, 0.8, 0xd49560);

            vPlatformEid = addStaticBox(state, -12, 2, 6, 4, 0.4, 4, 0x4078a0);
            state.addComponent(vPlatformEid, Move);
            addDynamicBox(state, -12, 2.7, 6, 0.8, 0xd49560);

            kLeverEid = addStaticBox(state, 8, 1, -18, 6, 0.6, 0.6, 0x4078a0);
            state.addComponent(kLeverEid, Move);
            addDynamicBox(state, 10, 0.5, -18, 0.8, 0xd49560);
            addDynamicBox(state, 6, 0.5, -18, 0.8, 0xd49560);

            addDynamicBox(state, 16, 0.5, -18, 0.8, 0xd49560);

            const heavy = state.addEntity();
            state.addComponent(heavy, Transform);
            state.addComponent(heavy, Part);
            state.addComponent(heavy, Body);
            Transform.posX[heavy] = 0;
            Transform.posY[heavy] = 1.25;
            Transform.posZ[heavy] = 5;
            Part.shape[heavy] = Shape.Box;
            Part.sizeX[heavy] = 2.5;
            Part.sizeY[heavy] = 2.5;
            Part.sizeZ[heavy] = 2.5;
            Part.color[heavy] = 0x3a3a3a;
            Body.mass[heavy] = 100;
            Body.friction[heavy] = 0.8;

            const anchor = state.addEntity();
            state.addComponent(anchor, Transform);
            state.addComponent(anchor, Part);
            state.addComponent(anchor, Body);
            Part.shape[anchor] = Shape.Sphere;
            Part.opacity[anchor] = 0;
            Body.mass[anchor] = 0;
            Transform.posY[anchor] = -1000;
            anchorEid = anchor;
        },
    };
}
