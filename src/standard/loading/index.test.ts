import { expect, test } from "bun:test";
import { build, UnsupportedError } from "../../engine";
import { minimalDark, minimalLight, shallotDark } from "./index";

class Element {
    style = { cssText: "", width: "", transition: "", opacity: "", background: "" };
    children: Element[] = [];
    textContent = "";
    innerHTML = "";
    isConnected = false;
    parentElement: Element | null = null;
    addEventListener(): void {}
    appendChild(child: Element): Element {
        this.children.push(child);
        child.parentElement = this;
        child.isConnected = true;
        return child;
    }
    insertBefore(child: Element, before: Element | null): Element {
        const index = before ? this.children.indexOf(before) : -1;
        if (index < 0) this.children.push(child);
        else this.children.splice(index, 0, child);
        child.isConnected = true;
        return child;
    }
    replaceChildren(...children: Element[]): void {
        this.children = children;
    }
    remove(): void {
        if (this.parentElement)
            this.parentElement.children = this.parentElement.children.filter(
                (child) => child !== this,
            );
        this.isConnected = false;
    }
}

function text(element: Element): string {
    return element.textContent + element.children.map(text).join("");
}

function contrast(foreground: string, background: string): number {
    const luminance = (hex: string) => {
        const channels = [1, 3, 5].map(
            (offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255,
        );
        const linear = channels.map((channel) =>
            channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4,
        );
        return 0.2126 * linear[0]! + 0.7152 * linear[1]! + 0.0722 * linear[2]!;
    };
    const light = Math.max(luminance(foreground), luminance(background));
    const dark = Math.min(luminance(foreground), luminance(background));
    return (light + 0.05) / (dark + 0.05);
}

function withDocument(run: (body: Element) => void): void {
    const previous = globalThis.document;
    const previousStyle = globalThis.getComputedStyle;
    const body = new Element();
    const fake = {
        body,
        createElement: () => new Element(),
        querySelector: () => null,
    } as unknown as Document;
    Object.defineProperty(globalThis, "document", { configurable: true, value: fake });
    Object.defineProperty(globalThis, "getComputedStyle", {
        configurable: true,
        value: () => ({ position: "static" }),
    });
    try {
        run(body);
    } finally {
        if (previous)
            Object.defineProperty(globalThis, "document", { configurable: true, value: previous });
        else Reflect.deleteProperty(globalThis, "document");
        if (previousStyle)
            Object.defineProperty(globalThis, "getComputedStyle", {
                configurable: true,
                value: previousStyle,
            });
        else Reflect.deleteProperty(globalThis, "getComputedStyle");
    }
}

test("minimal dark and light loading are transparent, preserve progress colors, readable notices and errors, and clean up; branded loading stays opaque", () => {
    withDocument((body) => {
        const themes = [
            {
                screen: minimalDark(),
                overlay: "#141210",
                track: "#1c1917",
                bar: "#d49560",
                muted: "#a08c78",
                surface: "#1c1917",
                text: "#f0e6d6",
                amber: "#d49560",
            },
            {
                screen: minimalLight(),
                overlay: "#f7f3ec",
                track: "#efe9df",
                bar: "#d49560",
                muted: "#6e655c",
                surface: "#efe9df",
                text: "#2a231e",
                amber: "#73512f",
            },
        ] as const;
        for (const theme of themes) {
            const cleanup = theme.screen.show();
            const overlay = body.children.at(-1)!;
            expect(overlay.style.cssText).toContain("background: transparent");
            theme.screen.update(0.4);
            const content = overlay.children[0]!;
            const track = content.children[0]!;
            expect(track.style.cssText).toContain(`background: ${theme.track}`);
            expect(track.children[0]!.style.cssText).toContain(`background: ${theme.bar}`);
            expect(track.children[0]!.style.width).toBe("40%");

            theme.screen.notice?.({ class: "fallback", identity: "test adapter" });
            const notice = content.children.at(-1)!;
            expect(notice.textContent).toBe("fallback adapter: test adapter");
            expect(notice.style.cssText).toContain(
                `color:${theme.muted};background:${theme.surface};`,
            );
            expect(contrast(theme.muted, theme.surface)).toBeGreaterThanOrEqual(4.5);

            theme.screen.error?.(new Error("scene preparation failed"));
            const card = overlay.children[0]!;
            expect(text(card)).toContain("scene preparation failed");
            expect(card.style.background).toBe(theme.overlay);
            expect(card.children[1]!.style.cssText).toContain(`color: ${theme.text};`);
            expect(contrast(theme.text, theme.overlay)).toBeGreaterThanOrEqual(4.5);

            theme.screen.error?.(
                new UnsupportedError("WebGPU is unavailable", ["timestamp-query"]),
            );
            const unsupported = overlay.children[0]!;
            expect(text(unsupported.children[1]!)).toContain("WebGPU is unavailable");
            expect(unsupported.style.background).toBe(theme.overlay);
            expect(unsupported.children[0]!.style.cssText).toContain(`color: ${theme.amber};`);
            expect(contrast(theme.amber, theme.overlay)).toBeGreaterThanOrEqual(4.5);
            cleanup?.();
            expect(body.children).toHaveLength(0);
        }
    });
});

test("build awaits an application-owned loading completion promise before cleanup and returning the app", async () => {
    let finish!: () => void;
    let entered!: () => void;
    let cleaned = false;
    let returned = false;
    const completionStarted = new Promise<void>((resolve) => (entered = resolve));
    const built = build({
        defaults: false,
        plugins: [],
        loading: {
            show: () => () => (cleaned = true),
            update: () => {},
            complete: () => {
                entered();
                return new Promise<void>((resolve) => (finish = resolve));
            },
        },
    }).then((app) => {
        returned = true;
        return app;
    });
    await completionStarted;
    const held = !returned && !cleaned;
    finish();
    const app = await built;
    try {
        expect(held).toBe(true);
        expect(cleaned).toBe(true);
    } finally {
        app.dispose();
    }
});

test("the shallot-branded dark loading screen keeps its established opaque background", () => {
    withDocument((body) => {
        shallotDark().show();
        expect(body.children[0]!.style.cssText).toContain("background: #141210");
    });
});
