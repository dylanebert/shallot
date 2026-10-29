import {
    Camera,
    Devices,
    mountOverlay,
    pressKey,
    releaseKey,
    releasePointerLock,
    requirePointerLock,
    Viewports,
    type World,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";

type Keys = Record<"W" | "A" | "S" | "D" | "Space" | "Shift", number>;
type Vector = { x: number; y: number; z: number };
const controls = {
    create: () => ({
        used: false,
        jumpPending: false,
        sampledTick: -1,
        buttons: [] as HTMLButtonElement[],
        stick: undefined as HTMLDivElement | undefined,
        knob: undefined as HTMLDivElement | undefined,
    }),
};

/** Samples movement and latched Jump once per frame; fixed gameplay consumes keys and the camera's horizontal basis. */
export function readPlayerControls(world: World, keys: Keys, forward: Vector, right: Vector): void {
    const devices = world.resource(Devices);
    const active = devices.enabled;
    const ui = world.resource(controls);
    const width = world.resource(Viewports).get(devices.focused)?.cssWidth ?? 0;
    let stick: import("@dylanebert/shallot").TouchPoint | undefined;
    let look: typeof stick;
    for (const point of devices.touch.held.values()) {
        if (point.startX < width / 2) stick ??= point;
        else look ??= point;
    }
    let x = 0,
        y = 0;
    if (active && stick) {
        const dx = stick.x - stick.startX,
            dy = stick.y - stick.startY;
        const magnitude = Math.hypot(dx, dy);
        if (magnitude > 7) {
            const scale = Math.min(1, magnitude / 70) / magnitude;
            x = dx * scale;
            y = -dy * scale;
        }
    }
    for (const key of ["W", "A", "S", "D", "Space", "Shift"] as const)
        keys[key] = Number(
            active &&
                (devices.keys.held.has(
                    key === "Shift" ? "ShiftLeft" : key === "Space" ? "Space" : `Key${key}`,
                ) ||
                    (key === "Shift" && devices.keys.held.has("ShiftRight"))),
        );
    if (ui.sampledTick !== world.time.fixedTick || !active || world.time.paused)
        ui.jumpPending = false;
    ui.sampledTick = world.time.fixedTick;
    if (active && !world.time.paused && devices.keys.pressed.has("Space")) ui.jumpPending = true;
    keys.Space = Number(!!keys.Space || ui.jumpPending);
    const lateral = Math.max(-1, Math.min(1, keys.D - keys.A + x));
    const advance = Math.max(-1, Math.min(1, keys.W - keys.S + y));
    keys.W = Math.max(0, advance);
    keys.S = Math.max(0, -advance);
    keys.D = Math.max(0, lateral);
    keys.A = Math.max(0, -lateral);
    if (devices.touch.pressed.size) ui.used = true;
    for (const button of ui.buttons) button.style.display = ui.used ? "" : "none";
    if (ui.stick && ui.knob) {
        ui.stick.style.display = active && stick ? "" : "none";
        if (stick) {
            ui.stick.style.left = `${stick.startX}px`;
            ui.stick.style.top = `${stick.startY}px`;
            ui.knob.style.transform = `translate(calc(-50% + ${x * 70}px), calc(-50% - ${y * 70}px))`;
        }
    }
    const [camera] = world.query([Camera, Orbit]);
    if (!camera) return;
    const orbit = world.storage(Orbit);
    let yaw = orbit.yaw.get(camera),
        pitch = orbit.pitch.get(camera);
    if (active) {
        if (look) {
            yaw -= (look.deltaX * 0.35 * Math.PI) / 180;
            pitch += (look.deltaY * 0.35 * Math.PI) / 180;
        } else if (!devices.touch.held.size && devices.pointer.lock.status === "locked") {
            yaw -= (devices.pointer.deltaX * 1.5) / 1080;
            pitch += (devices.pointer.deltaY * 1.5) / 1080;
        }
    }
    pitch = Math.max(-Math.PI * 0.45, Math.min(Math.PI * 0.45, pitch));
    orbit.yaw.set(camera, yaw);
    orbit.pitch.set(camera, pitch);
    // The camera's horizontal basis leaves vertical motion to the native character logic.
    Object.assign(forward, { x: -Math.sin(yaw), y: 0, z: -Math.cos(yaw) });
    Object.assign(right, { x: Math.cos(yaw), y: 0, z: -Math.sin(yaw) });
}

export function mountPlayerControls(world: World): () => void {
    requirePointerLock(world, true);
    const ui = world.resource(controls);
    let root: HTMLDivElement | undefined;
    if (typeof document !== "undefined") {
        root = document.createElement("div");
        root.className = "player-controls";
        mountOverlay(document.querySelector("canvas"), world).append(root);
        for (const [label, key, bottom] of [
            ["Jump", "Space", 104],
            ["Sprint", "ShiftLeft", 176],
        ] as const) {
            const button = document.createElement("button");
            button.tabIndex = -1;
            button.onfocus = () => button.blur();
            button.textContent = label;
            button.dataset.touchAction = "";
            button.style.cssText = `display:none;position:absolute;right:calc(24px + env(safe-area-inset-right));bottom:calc(${bottom}px + env(safe-area-inset-bottom));width:64px;height:64px;border-radius:50%;pointer-events:auto;touch-action:none`;
            button.addEventListener("pointerdown", (event) => {
                event.preventDefault();
                pressKey(world, key);
                try {
                    button.setPointerCapture(event.pointerId);
                } catch {
                    /* The press remains valid if its pointer has ended. */
                }
            });
            const release = () => releaseKey(world, key);
            button.addEventListener("pointerup", release);
            button.addEventListener("pointercancel", release);
            button.addEventListener("lostpointercapture", release);
            root.append(button);
            ui.buttons.push(button);
        }
        const stick = document.createElement("div");
        stick.dataset.touchStick = "";
        stick.style.cssText =
            "display:none;position:absolute;width:140px;height:140px;border:1px solid #2a2420;border-radius:50%;background:#141210cc;pointer-events:none;transform:translate(-50%,-50%)";
        const knob = document.createElement("div");
        knob.style.cssText =
            "position:absolute;left:50%;top:50%;width:48px;height:48px;border-radius:50%;background:#f0e6d680";
        stick.append(knob);
        root.append(stick);
        ui.stick = stick;
        ui.knob = knob;
    }
    return () => {
        root?.remove();
        ui.buttons.length = 0;
        ui.jumpPending = false;
        ui.sampledTick = -1;
        ui.stick = undefined;
        ui.knob = undefined;
        releaseKey(world, "Space");
        releaseKey(world, "ShiftLeft");
        requirePointerLock(world, false);
        releasePointerLock(world);
    };
}
