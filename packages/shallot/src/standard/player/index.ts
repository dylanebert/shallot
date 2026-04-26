import {
    traits,
    resource,
    pair,
    ChildOf,
    onRemove,
    Time,
    type State,
    type System,
    type Plugin,
} from "../../engine";
import { clamp } from "../../engine/utils";
import { Views } from "../viewport";
import { Transform } from "../transforms";
import { Inputs, InputPlugin } from "../input";
import { Character } from "../physics";
import { Camera } from "../render";

export const Player = {
    yaw: [] as number[],
    pitch: [] as number[],
    speed: [] as number[],
    sprint: [] as number[],
    sensitivity: [] as number[],
    eyeHeight: [] as number[],
    jumpBuffer: [] as number[],
};

traits(Player, {
    requires: [Transform],
    defaults: () => ({
        yaw: 0,
        pitch: 0,
        speed: 6,
        sprint: 1,
        sensitivity: 1.5,
        eyeHeight: 0.7,
        jumpBuffer: 0.1,
    }),
});

interface PointerLock {
    canvas: HTMLCanvasElement;
    locked: boolean;
    deltaX: number;
    deltaY: number;
    onClick: () => void;
    onChange: () => void;
    onMove: (e: MouseEvent) => void;
}

const PointerLock = resource<PointerLock>("pointerLock");

const HALF_PI = Math.PI / 2 - 0.01;

const cameras = new Map<number, number>();
const prev = new Map<number, Float64Array>();
const curr = new Map<number, Float64Array>();

export function playerCamera(eid: number): number | undefined {
    return cameras.get(eid);
}

function findCamera(state: State, eid: number): number {
    let cam = cameras.get(eid);
    if (cam !== undefined) return cam;

    for (const child of state.query([pair(ChildOf.relation, eid), Camera])) {
        cam = child;
        break;
    }

    if (cam === undefined) {
        console.warn(
            `Player entity ${eid} has Character but no Camera child. Add a camera entity as a child.`,
        );
        return -1;
    }

    cameras.set(eid, cam);
    const p = new Float64Array(3);
    const c = new Float64Array(3);
    p[0] = c[0] = Transform.posX[eid];
    p[1] = c[1] = Transform.posY[eid];
    p[2] = c[2] = Transform.posZ[eid];
    prev.set(eid, p);
    curr.set(eid, c);
    return cam;
}

const PlayerFixedSystem: System = {
    group: "fixed",
    last: true,

    update(state: State) {
        for (const eid of state.query([Player, Character, Transform])) {
            const p = prev.get(eid);
            const c = curr.get(eid);
            if (!p || !c) continue;
            p[0] = c[0];
            p[1] = c[1];
            p[2] = c[2];
            c[0] = Transform.posX[eid];
            c[1] = Transform.posY[eid];
            c[2] = Transform.posZ[eid];
        }
    },
};

const PlayerSystem: System = {
    group: "simulation",

    setup(state: State) {
        const views = Views.from(state);
        if (!views || views.size === 0) return;

        const canvas = views.values().next().value!.element;
        if (!canvas) return;
        const pl: PointerLock = {
            canvas,
            locked: false,
            deltaX: 0,
            deltaY: 0,
            onClick: () => {
                let hasPlayer = false;
                for (const _ of state.query([Player])) {
                    hasPlayer = true;
                    break;
                }
                if (hasPlayer) canvas.requestPointerLock().catch(() => {});
            },
            onChange: () => {
                pl.locked = document.pointerLockElement === canvas;
                if (pl.locked && document.activeElement instanceof HTMLElement) {
                    document.activeElement.blur();
                }
            },
            onMove: (e: MouseEvent) => {
                if (pl.locked) {
                    pl.deltaX += e.movementX;
                    pl.deltaY += e.movementY;
                }
            },
        };

        canvas.addEventListener("click", pl.onClick);
        document.addEventListener("pointerlockchange", pl.onChange);
        document.addEventListener("mousemove", pl.onMove);

        state.setResource(PointerLock, pl);

        state.observe(onRemove(Player), (eid) => {
            cameras.delete(eid);
            prev.delete(eid);
            curr.delete(eid);
        });
    },

    update(state: State) {
        const input = Inputs.from(state);
        const pl = PointerLock.from(state);
        if (!input || !pl) return;

        const dt = state.time.deltaTime;

        for (const eid of state.query([Player, Transform])) {
            const hasCharacter = state.hasComponent(eid, Character);

            if (pl.locked) {
                const scale = Player.sensitivity[eid] / pl.canvas.clientHeight;
                Player.yaw[eid] -= pl.deltaX * scale;
                Player.pitch[eid] = clamp(Player.pitch[eid] - pl.deltaY * scale, -HALF_PI, HALF_PI);
            }

            if (hasCharacter) {
                let lx = 0,
                    lz = 0;
                if (input.isKeyDown("KeyW")) lz -= 1;
                if (input.isKeyDown("KeyS")) lz += 1;
                if (input.isKeyDown("KeyA")) lx -= 1;
                if (input.isKeyDown("KeyD")) lx += 1;

                const len = Math.sqrt(lx * lx + lz * lz);
                if (len > 0) {
                    const sprinting = input.isKeyDown("ShiftLeft") || input.isKeyDown("ShiftRight");
                    const speed =
                        (Character.speed[eid] * (sprinting ? Player.sprint[eid] : 1)) / len;
                    const yaw = Player.yaw[eid];
                    const cy = Math.cos(yaw),
                        sy = Math.sin(yaw);
                    Character.moveX[eid] = (lz * sy + lx * cy) * speed;
                    Character.moveZ[eid] = (lz * cy - lx * sy) * speed;
                } else {
                    Character.moveX[eid] = 0;
                    Character.moveZ[eid] = 0;
                }

                Character.jump[eid] =
                    input.isKeyDown("Space") ||
                    input.isKeyPressedWithin("Space", Player.jumpBuffer[eid])
                        ? 1
                        : 0;

                const cam = findCamera(state, eid);
                if (cam < 0) continue;

                const p = prev.get(eid);
                const c = curr.get(eid);
                if (p && c) {
                    const alpha = Math.min(state.scheduler.accumulator / Time.FIXED_DT, 1);
                    const bx = Transform.posX[eid];
                    const by = Transform.posY[eid];
                    const bz = Transform.posZ[eid];
                    Transform.posX[cam] = p[0] + (c[0] - p[0]) * alpha - bx;
                    Transform.posY[cam] = p[1] + (c[1] - p[1]) * alpha - by + Player.eyeHeight[eid];
                    Transform.posZ[cam] = p[2] + (c[2] - p[2]) * alpha - bz;
                }

                const hy = Player.yaw[eid] * 0.5;
                const hp = Player.pitch[eid] * 0.5;
                const sy = Math.sin(hy),
                    cy = Math.cos(hy);
                const sp = Math.sin(hp),
                    cp = Math.cos(hp);
                Transform.quatX[cam] = cy * sp;
                Transform.quatY[cam] = sy * cp;
                Transform.quatZ[cam] = -sy * sp;
                Transform.quatW[cam] = cy * cp;
            } else {
                if (pl.locked) {
                    let mx = 0,
                        mz = 0;
                    if (input.isKeyDown("KeyW")) mz -= 1;
                    if (input.isKeyDown("KeyS")) mz += 1;
                    if (input.isKeyDown("KeyA")) mx -= 1;
                    if (input.isKeyDown("KeyD")) mx += 1;

                    const sprinting = input.isKeyDown("ShiftLeft") || input.isKeyDown("ShiftRight");
                    const speed = Player.speed[eid] * 0.5 * (sprinting ? Player.sprint[eid] : 1);

                    const len = Math.sqrt(mx * mx + mz * mz);
                    if (len > 0) {
                        const move = (speed * dt) / len;
                        const yaw = Player.yaw[eid];
                        const cy = Math.cos(yaw),
                            sy = Math.sin(yaw);
                        Transform.posX[eid] += (mz * sy + mx * cy) * move;
                        Transform.posZ[eid] += (mz * cy - mx * sy) * move;
                    }
                }

                const hy = Player.yaw[eid] * 0.5;
                const hp = Player.pitch[eid] * 0.5;
                const sy = Math.sin(hy),
                    cy = Math.cos(hy);
                const sp = Math.sin(hp),
                    cp = Math.cos(hp);
                Transform.quatX[eid] = cy * sp;
                Transform.quatY[eid] = sy * cp;
                Transform.quatZ[eid] = -sy * sp;
                Transform.quatW[eid] = cy * cp;
            }
        }

        pl.deltaX = 0;
        pl.deltaY = 0;
    },

    dispose(state: State) {
        cameras.clear();
        prev.clear();
        curr.clear();

        const pl = PointerLock.from(state);
        if (!pl) return;

        if (pl.locked) document.exitPointerLock();
        pl.canvas.removeEventListener("click", pl.onClick);
        document.removeEventListener("pointerlockchange", pl.onChange);
        document.removeEventListener("mousemove", pl.onMove);
        state.deleteResource(PointerLock);
    },
};

export const PlayerPlugin: Plugin = {
    name: "Player",
    systems: [PlayerFixedSystem, PlayerSystem],
    components: { Player },
    dependencies: [InputPlugin],
};
