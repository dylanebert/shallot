import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import {
    DARK,
    END_TICK,
    fromBlocks,
    type Grid,
    HIT_TICK,
    lockup,
    MARK,
    progressTick,
    splash,
    splashFrame,
    TICK_MS,
    toAnsi,
    toCells,
    toSvg,
    toText,
    word,
} from "./mark";

describe("mark", () => {
    test("half blocks round-trip through the bitmap", () => {
        for (const rows of Object.values(MARK)) {
            expect(toText(toCells(fromBlocks(rows)))).toBe(rows.join("\n"));
        }
    });

    test("the canonical mark is 12 by 14 with a one-row tip", () => {
        const grid = fromBlocks(MARK.m);
        expect(grid.length).toBe(14);
        expect(grid[0]?.length).toBe(12);
        expect(grid[0]?.filter(Boolean).length).toBe(0);
        expect(grid[1]?.filter(Boolean).length).toBe(2);
    });

    test("the wordmark is 36 by 7 in ink", () => {
        const grid = word();
        expect(grid.length).toBe(7);
        expect(grid[0]?.length).toBe(36);
        expect(new Set(grid.flat().filter(Boolean))).toEqual(new Set(["ink"]));
    });

    test("svg emits one rect per pixel at the given scale", () => {
        const grid = fromBlocks(MARK.s);
        const pixels = grid.flat().filter(Boolean).length;
        const svg = toSvg(grid, DARK, 3);
        expect(svg.match(/<rect /g)?.length).toBe(pixels);
        expect(svg).toContain('width="30" height="36"');
    });

    test("ansi resets at every row end", () => {
        for (const line of toAnsi(toCells(lockup())).split("\n")) {
            if (line.includes("\x1b[38")) expect(line.endsWith("\x1b[0m")).toBe(true);
        }
    });
});

describe("splash", () => {
    test("ends on the lockup with no cursor", () => {
        expect(splashFrame(END_TICK + 1)).toEqual(lockup());
        expect(splashFrame(END_TICK + 30)).toEqual(lockup());
    });

    test("the mark is complete at the hit and the name is not", () => {
        const at = splashFrame(HIT_TICK);
        const mark = fromBlocks(MARK.m);
        mark.forEach((row, y) => row.forEach((tone, x) => tone && expect(at[y]?.[x]).toBeTruthy()));
        expect(at.flat().filter((t) => t === "ink").length).toBe(0);
    });

    test("pixels only ever switch on, never off, before the hit", () => {
        let lit = 0;
        for (let t = 0; t <= HIT_TICK; t++) {
            const now = splashFrame(t)
                .flat()
                .filter((v) => v === "gold" || v === "dim").length;
            expect(now).toBeGreaterThanOrEqual(lit);
            lit = now;
        }
    });

    test("no frame carries a colour outside the three tones", () => {
        for (let t = 0; t <= END_TICK + 1; t++) {
            for (const v of splashFrame(t).flat())
                expect([null, "gold", "dim", "ink"]).toContain(v);
        }
    });
});

describe("progressTick", () => {
    test("spans zero to the hit and never falls back across the range", () => {
        expect(progressTick(0)).toBe(0);
        expect(progressTick(1)).toBe(HIT_TICK);
        let last = 0;
        for (let i = 0; i < 1000; i++) {
            const tick = progressTick(i / 1000);
            expect(tick).toBeLessThan(HIT_TICK);
            expect(tick).toBeGreaterThanOrEqual(last);
            last = tick;
        }
    });

    test("clamps out-of-range progress and reads a non-finite one as zero", () => {
        expect(progressTick(-1)).toBe(0);
        expect(progressTick(2)).toBe(HIT_TICK);
        expect(progressTick(Number.NaN)).toBe(0);
        expect(progressTick(Number.POSITIVE_INFINITY)).toBe(0);
    });

    test("progress lights the mark and never the name", () => {
        const lit = (grid: Grid) => grid.flat().filter((t) => t === "gold" || t === "dim").length;
        let last = 0;
        for (let i = 0; i <= 100; i++) {
            const frame = splashFrame(progressTick(i / 100));
            expect(lit(frame)).toBeGreaterThanOrEqual(last);
            last = lit(frame);
            expect(frame.flat().filter((t) => t === "ink").length).toBe(0);
        }
        expect(lit(splashFrame(progressTick(0.99)))).toBeLessThan(
            lit(splashFrame(progressTick(1))),
        );
    });
});

describe("splash driver", () => {
    let el: { innerHTML: string };
    let now: number;
    let queue: FrameRequestCallback[];
    let nowSpy: ReturnType<typeof spyOn>;
    const render = (grid: Grid) => toSvg(grid, DARK, 1);
    // a hair past the tick boundary, so floating-point division never lands a frame a tick short
    const advance = () => {
        now += TICK_MS + 0.001;
    };
    const frame = () => {
        const next = queue.shift();
        next?.(now);
    };

    beforeEach(() => {
        el = { innerHTML: "" };
        now = 0;
        queue = [];
        nowSpy = spyOn(performance, "now").mockImplementation(() => now);
        (globalThis as any).requestAnimationFrame = (cb: FrameRequestCallback) => queue.push(cb);
        (globalThis as any).cancelAnimationFrame = () => {
            queue.length = 0;
        };
    });

    afterEach(() => {
        nowSpy.mockRestore();
        delete (globalThis as any).requestAnimationFrame;
        delete (globalThis as any).cancelAnimationFrame;
    });

    test("seek draws a tick and skips a repeat", () => {
        const driver = splash(el as unknown as Element, render);
        driver.seek(HIT_TICK);
        expect(el.innerHTML).toBe(render(splashFrame(HIT_TICK)));
        el.innerHTML = "stale";
        driver.seek(HIT_TICK);
        expect(el.innerHTML).toBe("stale");
        expect(queue.length).toBe(0);
    });

    test("play advances one tick per TICK_MS and resolves once the lockup is drawn", async () => {
        const driver = splash(el as unknown as Element, render);
        let done = false;
        const playing = driver.play(HIT_TICK).then(() => {
            done = true;
        });
        for (let i = 0; i < END_TICK - HIT_TICK; i++) {
            advance();
            frame();
        }
        await Promise.resolve();
        expect(done).toBe(false);
        expect(el.innerHTML).toBe(render(splashFrame(END_TICK)));
        advance();
        frame();
        await playing;
        expect(done).toBe(true);
        expect(el.innerHTML).toBe(render(lockup()));
    });

    test("a second play resolves the first and restarts the clock", async () => {
        const driver = splash(el as unknown as Element, render);
        let first = false;
        driver.play(HIT_TICK).then(() => {
            first = true;
        });
        advance();
        frame();
        driver.play();
        await Promise.resolve();
        expect(first).toBe(true);
        expect(el.innerHTML).toBe(render(splashFrame(0)));
    });

    test("reduced motion rests on the lockup and resolves with no frame queued", async () => {
        const driver = splash(el as unknown as Element, render, true);
        driver.seek(0);
        expect(el.innerHTML).toBe(render(lockup()));
        await driver.play(HIT_TICK);
        expect(queue.length).toBe(0);
    });
});
