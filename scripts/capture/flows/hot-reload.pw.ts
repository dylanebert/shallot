import { type Page, test } from "@playwright/test";
import { defineFlow } from "../runner";
import { S } from "../selectors";

const PORT = process.env.CAPTURE_PORT || "3004";
const API = `http://localhost:${PORT}/__api`;
const FILE = "src/ticker.ts";

// the end-to-end check of the editor hot-reload wiring (testing.md "Reload tier"): the engine
// `swap` is bun-test-covered, but the live virtual:project HMR accept → emitProjectReload → swap path
// runs only in the editor. Four phases over one session, covering the live swap-vs-rebuild matrix:
//   1. edit-mode swap — a constant edit lands as an IN-PLACE swap, not a rebuild: the behavior the
//      system writes (`mark`) changes live while the runtime counter (`ticks`) keeps climbing
//   2. play-mode swap — the same edit against the play State (same onProjectReload → swap path)
//   3. rebuild fallback — a component-schema edit is rejected by `swap` and the editor rebuilds from
//      the document: the new schema goes live, the editor stays responsive, the counter resets
//   4. failure recovery — a reloaded system that throws is quarantined (the frame loop and control
//      seam stay live, the counter freezes), and the next good edit swaps in and resumes it
// Edits go through the dev server's /__api/file seam so the write lands where the watcher watches.

interface TickerRead {
    ticks: number;
    mark: number;
    extra?: number;
}

// the warm-spawned ticker entity's fields, read through the control seam (TickSystem keeps it written).
async function ticker(page: Page): Promise<TickerRead | null> {
    const res = await page.request.get(`${API}/entities?component=ticker`);
    const ents = (await res.json()) as { components: { ticker?: TickerRead } }[];
    return ents[0]?.components?.ticker ?? null;
}

async function poll(
    page: Page,
    want: (t: TickerRead | null) => boolean,
    timeoutMs = 10_000,
): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (want(await ticker(page))) return true;
        await page.waitForTimeout(150);
    }
    return false;
}

defineFlow({ name: "hot-reload", scene: "demo" }, async ({ page, step, act, assert }) => {
    // each HMR roundtrip (write → watcher → recompile → swap or rebuild) costs seconds; three
    // phases hold six of them
    test.setTimeout(120_000);

    const original = await (await page.request.get(`${API}/file?path=${FILE}`)).text();
    const m = original.match(/const MARK = (\d+);/);
    assert("fixture exposes the MARK constant", m !== null);
    const oldMark = Number(m![1]);
    const newMark = oldMark + 1;
    const write = (content: string) =>
        page.request.post(`${API}/file?path=${FILE}`, { data: { content } });

    // phase 1 — edit-mode swap: the edited behavior goes live, the runtime counter survives
    try {
        await page.waitForTimeout(500);
        const before = await ticker(page);
        assert("ticker entity exists in edit mode", before !== null);
        assert(
            `ticker writes the current MARK (${oldMark}, got ${before!.mark})`,
            before!.mark === oldMark,
        );
        assert(`ticker counter is climbing (ticks ${before!.ticks})`, before!.ticks > 0);

        await write(original.replace(/const MARK = \d+;/, `const MARK = ${newMark};`));
        assert(
            `the edited behavior went live (mark → ${newMark})`,
            await poll(page, (t) => t?.mark === newMark),
        );

        const after = await ticker(page);
        assert(
            `the runtime counter survived the swap, no rebuild (before ${before!.ticks}, after ${after!.ticks})`,
            after!.ticks >= before!.ticks,
        );
        await step("reloaded", { highlight: [S.viewport] });
    } finally {
        await write(original);
        await poll(page, (t) => t?.mark === oldMark, 8000);
    }

    // phase 2 — play-mode swap: the same onProjectReload → swap path against the play State
    await act.click(S.playBtn);
    assert("play State runs the ticker", await poll(page, (t) => t !== null && t.ticks > 0));
    try {
        const before = await ticker(page);
        await write(original.replace(/const MARK = \d+;/, `const MARK = ${newMark};`));
        assert(
            `the edit swapped onto the play State (mark → ${newMark})`,
            await poll(page, (t) => t?.mark === newMark),
        );
        const after = await ticker(page);
        assert(
            `play-mode runtime state survived the swap (before ${before!.ticks}, after ${after!.ticks})`,
            after!.ticks >= before!.ticks,
        );
        await step("play-reloaded", { highlight: [S.viewport] });
    } finally {
        await write(original);
        await poll(page, (t) => t?.mark === oldMark, 8000);
        await act.click(S.stopBtn);
        await act.wait(500);
    }

    // phase 3 — rebuild fallback: a schema edit is rejected by swap; the editor rebuilds from the
    // document (warm re-runs, counter resets) and the new schema goes live, never wedging
    try {
        const schemaEdit = original.replace(
            "mark: sparse(u32) }",
            "mark: sparse(u32), extra: sparse(u32) }",
        );
        assert("fixture exposes the Ticker schema", schemaEdit !== original);
        await write(schemaEdit);
        assert(
            "the schema edit forced a rebuild and the new field went live",
            await poll(page, (t) => t != null && t.extra !== undefined, 15_000),
        );
        assert(
            "the rebuilt State runs (counter climbing, behavior intact)",
            await poll(page, (t) => t != null && t.ticks > 0 && t.mark === oldMark),
        );
        await step("rebuilt", { highlight: [S.viewport] });
    } finally {
        await write(original);
        await poll(page, (t) => t != null && t.extra === undefined, 15_000);
    }

    // phase 4 — failure recovery: the throwing update swaps in (same shape), throws once, and is
    // quarantined; the editor stays responsive (the control seam still answers, no wedged frame
    // loop), and the next good edit swaps onto the same slot and the system resumes
    try {
        await poll(page, (t) => t !== null && t.ticks > 0);
        const throwEdit = original.replace(
            "Ticker.mark.set(eid, MARK);",
            'throw new Error("ticker boom");',
        );
        assert("fixture exposes the mark write to break", throwEdit !== original);
        await write(throwEdit);

        // quarantined: the counter freezes while the editor keeps serving requests
        let frozen = false;
        const deadline = Date.now() + 15_000;
        while (Date.now() < deadline && !frozen) {
            const a = await ticker(page);
            await page.waitForTimeout(400);
            const b = await ticker(page);
            frozen = a !== null && b !== null && a.ticks === b.ticks;
        }
        assert("the throwing system was paused (counter frozen, editor responsive)", frozen);

        await write(original.replace(/const MARK = \d+;/, `const MARK = ${newMark};`));
        assert(
            `the next good edit resumed the system (mark → ${newMark})`,
            await poll(page, (t) => t?.mark === newMark),
        );
        const resumed = await ticker(page);
        const resumedAgain = await poll(page, (t) => t !== null && t.ticks > resumed!.ticks);
        assert("the counter climbs again after recovery", resumedAgain);
        await step("recovered", { highlight: [S.viewport] });
    } finally {
        await write(original);
        await poll(page, (t) => t?.mark === oldMark, 8000);
    }
});
