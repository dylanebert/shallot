import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { minimalDark, shallotDark } from "./";
import {
    DARK,
    END_TICK,
    HIT_TICK,
    lockup,
    progressTick,
    splashFrame,
    TICK_MS,
    toSvg,
} from "./mark";

function createMockElement(tag: string) {
    const el: Record<string, any> = {
        tagName: tag.toUpperCase(),
        style: { cssText: "", width: "", position: "" },
        children: [] as any[],
        innerHTML: "",
        appendChild(child: any) {
            el.children.push(child);
            return child;
        },
        remove() {},
    };
    return el;
}

describe("Loading", () => {
    describe("factory exports", () => {
        test("shallotDark returns Loading interface", () => {
            const loading = shallotDark();
            expect(typeof loading.show).toBe("function");
            expect(typeof loading.update).toBe("function");
        });

        test("minimalDark returns Loading interface", () => {
            const loading = minimalDark();
            expect(typeof loading.show).toBe("function");
            expect(typeof loading.update).toBe("function");
        });
    });

    describe("headless environment", () => {
        test("show returns void when document is undefined", () => {
            const loading = shallotDark();
            const cleanup = loading.show();
            expect(cleanup).toBeUndefined();
        });

        test("complete resolves at once with no splash to play", async () => {
            const loading = shallotDark();
            loading.show();
            await loading.complete?.();
        });
        // the null-bar no-op path `update()` takes here (no overlay, no bar) is positively
        // asserted by "update after cleanup leaves the bar untouched" in the DOM block — same branch
    });

    describe("DOM lifecycle", () => {
        let mockBody: Record<string, any>;
        let createdElements: Record<string, any>[];
        let frames: number;
        let reduced: boolean;
        let now: number;
        let queue: FrameRequestCallback[];
        let nowSpy: ReturnType<typeof spyOn>;

        beforeEach(() => {
            createdElements = [];
            frames = 0;
            reduced = false;
            now = 0;
            queue = [];
            nowSpy = spyOn(performance, "now").mockImplementation(() => now);
            mockBody = createMockElement("body");
            mockBody.style.position = "static";

            const mockDoc = {
                createElement(tag: string) {
                    const el = createMockElement(tag);
                    createdElements.push(el);
                    return el;
                },
                querySelector() {
                    return null;
                },
                body: mockBody,
            };
            (globalThis as any).document = mockDoc;
            (globalThis as any).getComputedStyle = () => ({ position: "static" });
            (globalThis as any).matchMedia = () => ({ matches: reduced });
            // queued, drained a frame at a time: the tests that run the outro clock drive them
            (globalThis as any).requestAnimationFrame = (cb: FrameRequestCallback) => {
                queue.push(cb);
                return ++frames;
            };
            (globalThis as any).cancelAnimationFrame = () => {
                queue.length = 0;
            };
        });

        afterEach(() => {
            nowSpy.mockRestore();
            delete (globalThis as any).document;
            delete (globalThis as any).getComputedStyle;
            delete (globalThis as any).matchMedia;
            delete (globalThis as any).requestAnimationFrame;
            delete (globalThis as any).cancelAnimationFrame;
        });

        test("show creates overlay and returns cleanup", () => {
            const loading = shallotDark();
            const cleanup = loading.show();
            expect(typeof cleanup).toBe("function");
            expect(mockBody.children.length).toBe(1);
        });

        test("cleanup removes overlay", () => {
            const loading = shallotDark();
            const cleanup = loading.show()!;
            let removed = false;
            mockBody.children[0].remove = () => {
                removed = true;
            };
            cleanup();
            expect(removed).toBe(true);
        });

        test("update sets bar width percentage", () => {
            const loading = shallotDark();
            loading.show();
            loading.update(0.5);
            const bar = createdElements[4];
            expect(bar.style.width).toBe("50%");
        });

        test("update after cleanup leaves the bar untouched", () => {
            const loading = shallotDark();
            const cleanup = loading.show()!;
            loading.update(0.5);
            const bar = createdElements[4]; // overlay, panel, splash, track, bar
            expect(bar.style.width).toBe("50%");

            cleanup(); // nulls the bar ref
            loading.update(0.75); // no-op: nothing to write to
            expect(bar.style.width).toBe("50%"); // detached bar unchanged, not "75%"
        });

        test("shallot variant mounts the splash and the track at rest", () => {
            const loading = shallotDark();
            loading.show();
            const panel = mockBody.children[0].children[0]; // overlay → centered panel
            expect(panel.children.length).toBe(2);
            // progress drives the landing, so `show` draws tick zero and starts no clock
            expect(panel.children[0].innerHTML).toBe(toSvg(splashFrame(0), DARK, 4));
            expect(frames).toBe(0);
        });

        test("update seeks the splash to the progress tick and never back", () => {
            const loading = shallotDark();
            loading.show();
            const splash = mockBody.children[0].children[0].children[0];
            const bar = createdElements[4];
            loading.update(0.5);
            expect(bar.style.width).toBe("50%");
            expect(splash.innerHTML).toBe(toSvg(splashFrame(progressTick(0.5)), DARK, 4));
            loading.update(0.6);
            loading.update(0.3);
            expect(splash.innerHTML).toBe(toSvg(splashFrame(progressTick(0.6)), DARK, 4));
            expect(frames).toBe(0);
        });

        test("minimal variants carry no complete", () => {
            expect(minimalDark().complete).toBeUndefined();
            expect(shallotDark().complete).toBeDefined();
        });

        test("complete holds until the outro reaches the lockup", async () => {
            const loading = shallotDark();
            loading.show();
            const splash = mockBody.children[0].children[0].children[0];
            loading.update(1);
            let done = false;
            const held = Promise.resolve(loading.complete?.()).then(() => {
                done = true;
            });
            for (let i = 0; i < END_TICK - HIT_TICK; i++) {
                now += TICK_MS + 0.001;
                queue.shift()?.(now);
            }
            await Promise.resolve();
            expect(done).toBe(false);
            now += TICK_MS + 0.001;
            queue.shift()?.(now);
            await held;
            expect(splash.innerHTML).toBe(toSvg(lockup(), DARK, 4));
        });

        test("reduced-motion complete resolves with no frame queued", async () => {
            reduced = true;
            const loading = shallotDark();
            loading.show();
            await loading.complete?.();
            expect(frames).toBe(0);
        });

        test("complete after cleanup resolves at once", async () => {
            const loading = shallotDark();
            const cleanup = loading.show()!;
            cleanup();
            await loading.complete?.();
            expect(frames).toBe(0);
        });

        test("reduced motion renders the resting lockup once", () => {
            reduced = true;
            const loading = shallotDark();
            loading.show();
            const splash = mockBody.children[0].children[0].children[0];
            expect(splash.innerHTML).toBe(toSvg(lockup(), DARK, 4));
            expect(frames).toBe(0);
        });

        test("minimal variant has track only", () => {
            const loading = minimalDark();
            loading.show();
            const panel = mockBody.children[0].children[0]; // overlay → centered panel
            expect(panel.children.length).toBe(1);
        });

        test("minimal update sets bar width", () => {
            const loading = minimalDark();
            loading.show();
            loading.update(1.0);
            const bar = createdElements[3]; // overlay, panel, track, bar
            expect(bar.style.width).toBe("100%");
        });

        test("sets parent position to relative when static", () => {
            const loading = shallotDark();
            loading.show();
            expect(mockBody.style.position).toBe("relative");
        });
    });
});
