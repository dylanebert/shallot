import { expect, test } from "bun:test";
import { createBrowserInputPlugin, Devices, focusCanvas, InputPlugin } from "../../core/input";
import { World } from "../../engine";
import { DiagnosticsOverlayPlugin, registerDebugChord, showProfiler } from ".";

function fixture() {
    const world = new World();
    const capture = new EventTarget();
    const bubble = new EventTarget();
    // Model window capture before window bubble, even when input attaches first.
    const target = {
        addEventListener(
            type: string,
            listener: EventListener,
            options?: AddEventListenerOptions | boolean,
        ) {
            const capturing = typeof options === "boolean" ? options : options?.capture;
            (capturing ? capture : bubble).addEventListener(type, listener, options);
        },
        removeEventListener(
            type: string,
            listener: EventListener,
            options?: EventListenerOptions | boolean,
        ) {
            const capturing = typeof options === "boolean" ? options : options?.capture;
            (capturing ? capture : bubble).removeEventListener(type, listener, options);
        },
    };
    const canvas = Object.assign(new EventTarget(), { style: { touchAction: "auto" } });
    const input = createBrowserInputPlugin({
        window: target as unknown as Window,
        document: new EventTarget() as Document,
        queryCanvases: () => [canvas as unknown as HTMLCanvasElement],
        supportsPointerLock: () => false,
        requestPointerLock() {},
        releasePointerLock() {},
    });
    for (const plugin of [InputPlugin, input]) {
        for (const system of plugin.systems ?? []) world.addSystem(system, plugin.name);
    }
    bubble.addEventListener("keydown", (event) => event.preventDefault());
    world.step(0);
    focusCanvas(world, 0);
    const previous = globalThis.window;
    globalThis.window = target as unknown as Window & typeof globalThis;
    try {
        DiagnosticsOverlayPlugin.initialize!(world);
    } finally {
        if (previous === undefined) Reflect.deleteProperty(globalThis, "window");
        else globalThis.window = previous;
    }
    const key = (type: string, code: string, repeat = false) => {
        const event = Object.assign(new Event(type, { cancelable: true }), { code, repeat });
        let stopped = false;
        event.stopPropagation = () => {
            stopped = true;
        };
        capture.dispatchEvent(event);
        if (!stopped) bubble.dispatchEvent(event);
    };
    return { world, key };
}

test("F3 toggles once on release, not on press or repeat", () => {
    const { world, key } = fixture();
    try {
        key("keydown", "F3");
        expect(showProfiler(world, false)).toBe(false);
        key("keydown", "F3", true);
        key("keyup", "F3");
        expect(showProfiler(world, false)).toBe(true);
        key("keyup", "F3");
        expect(showProfiler(world, false)).toBe(false);
    } finally {
        world.dispose();
    }
});

test("F3+R runs once, preserves visibility and never delivers held R to World input", () => {
    const { world, key } = fixture();
    try {
        let calls = 0;
        registerDebugChord(world, "KeyR", () => calls++);
        showProfiler(world);
        key("keydown", "F3");
        key("keydown", "KeyR");
        key("keydown", "KeyR", true);
        key("keyup", "F3");
        key("keydown", "KeyR", true);
        expect(world.resource(Devices).keys.held.has("KeyR")).toBe(false);
        key("keyup", "KeyR");
        expect(world.resource(Devices).keys.released.has("KeyR")).toBe(false);
        expect(calls).toBe(1);
        expect(showProfiler(world, false)).toBe(true);
    } finally {
        world.dispose();
    }
});

test("unregistered F3+K suppresses the toggle but still delivers K", () => {
    const { world, key } = fixture();
    try {
        key("keydown", "F3");
        key("keydown", "KeyK");
        key("keyup", "F3");
        expect(showProfiler(world, false)).toBe(false);
        expect(world.resource(Devices).keys.held.has("KeyK")).toBe(true);
    } finally {
        world.dispose();
    }
});

test("preventing a browser default still delivers Space to World input", () => {
    const { world, key } = fixture();
    try {
        key("keydown", "Space");
        expect(world.resource(Devices).keys.held.has("Space")).toBe(true);
    } finally {
        world.dispose();
    }
});
