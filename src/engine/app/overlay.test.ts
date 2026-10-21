import { expect, test } from "bun:test";
import { World } from "../ecs";
import { mountOverlay, OverlayCanvases } from ".";

interface FixtureElement {
    style: { position: string; cssText: string };
    parentElement: FixtureElement | null;
    appendChild(child: FixtureElement): void;
    remove(): void;
}

function element(): FixtureElement {
    return {
        style: { position: "", cssText: "" },
        parentElement: null as FixtureElement | null,
        appendChild(child: FixtureElement) {
            child.parentElement = this;
        },
        remove() {
            this.parentElement = null;
        },
    };
}

test("mountOverlay registers canvas ownership per world and releases it on disposal", () => {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, "document");
    const body = element();
    Object.defineProperty(globalThis, "document", {
        configurable: true,
        value: { body, createElement: element },
    });
    const world = new World();
    const other = new World();
    try {
        const parent = element();
        parent.style.position = "sticky";
        const canvas = element();
        parent.appendChild(canvas);
        const overlay = mountOverlay(canvas as unknown as HTMLElement, world);
        expect(world.resource(OverlayCanvases).get(overlay)).toBe(canvas as unknown as HTMLElement);
        expect(other.resource(OverlayCanvases).size).toBe(0);
        expect(overlay.parentElement).toBe(parent as unknown as HTMLElement);
        expect(parent.style.position).toBe("relative");
        const ownership = world.resource(OverlayCanvases);
        world.dispose();
        expect(ownership.size).toBe(0);
        expect(overlay.parentElement).toBeNull();
        expect(parent.style.position).toBe("sticky");
    } finally {
        world.dispose();
        other.dispose();
        if (descriptor) Object.defineProperty(globalThis, "document", descriptor);
        else Reflect.deleteProperty(globalThis, "document");
    }
});

test("mountOverlay without a canvas or world does not register ownership", () => {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, "document");
    const body = element();
    Object.defineProperty(globalThis, "document", {
        configurable: true,
        value: { body, createElement: element },
    });
    const world = new World();
    try {
        const overlay = mountOverlay(null, world);
        expect(overlay.parentElement).toBe(body as unknown as HTMLElement);
        expect(world.resource(OverlayCanvases).size).toBe(0);
        const canvas = element();
        body.appendChild(canvas);
        const unmanaged = mountOverlay(canvas as unknown as HTMLElement);
        world.dispose();
        expect(overlay.parentElement).toBeNull();
        expect(unmanaged.parentElement).toBe(body as unknown as HTMLElement);
        unmanaged.remove();
    } finally {
        world.dispose();
        if (descriptor) Object.defineProperty(globalThis, "document", descriptor);
        else Reflect.deleteProperty(globalThis, "document");
    }
});
