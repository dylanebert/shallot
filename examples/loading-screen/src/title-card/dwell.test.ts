import { expect, test } from "bun:test";
import { completeIntro, MINIMUM_DWELL_MS } from "./dwell";
import { titleCard } from "./index";

test("the title card exposes the completion hold to the engine", () => {
    expect(typeof titleCard({} as HTMLElement).complete).toBe("function");
});

test("completion waits for the intro and the remaining dwell measured from show", async () => {
    let end!: () => void;
    const intro = new Promise<void>((resolve) => {
        end = resolve;
    });
    const waits: number[] = [];
    let finished = false;
    const completion = completeIntro(
        100,
        intro,
        () => 600,
        async (ms) => {
            waits.push(ms);
        },
    ).then(() => {
        finished = true;
    });
    await Promise.resolve();
    expect(finished).toBe(false);
    expect(waits).toEqual([]);
    end();
    await completion;
    expect(waits).toEqual([MINIMUM_DWELL_MS - 500]);
});

test("an intro ending after the dwell adds no further hold", async () => {
    const waits: number[] = [];
    await completeIntro(
        100,
        Promise.resolve(),
        () => 2100,
        async (ms) => {
            waits.push(ms);
        },
    );
    expect(waits).toEqual([0]);
});
