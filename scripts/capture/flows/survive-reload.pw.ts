import { type Page, test } from "@playwright/test";
import { defineFlow } from "../runner";
import { S } from "../selectors";

// Stage-7 survive-reload e2e (testing.md "Reload tier"): a standalone run({survive}) app — the real
// production path the engine bun-tests can't reach, not the editor. The fixture's `Counter.n` climbs
// every frame; on a REAL page.reload() the beforeunload→sessionStorage→restore-on-boot path must bring
// it back (a fresh build would reset it to the scene default, 0). The warm-derived Sprout must NOT
// double — `serialize` captures only the authored set, so a restore (load + warm) re-derives exactly one.

type Survive = { n: number; sprouts: number };

const read = (page: Page): Promise<Survive | null> =>
    page.evaluate(() => (window as Window & { __survive?: () => Survive }).__survive?.() ?? null);

defineFlow(
    { name: "survive-reload", scene: "survive", target: "app" },
    async ({ page, step, assert }) => {
        test.setTimeout(30_000);

        // let Counter.n climb well past the scene default (0), so a restore is distinguishable from a fresh boot
        await page.waitForFunction(
            () => ((window as Window & { __survive?: () => Survive }).__survive?.()?.n ?? 0) > 20,
            null,
            { timeout: 8000 },
        );
        const before = await read(page);
        assert("the standalone app exposes the survive seam", before !== null);
        assert(`counter climbed in the standalone app (n ${before!.n})`, before!.n > 20);
        assert(`exactly one warm-derived sprout (got ${before!.sprouts})`, before!.sprouts === 1);

        // a real reload: beforeunload serializes the live State, boot restores it in place of the scene
        await page.reload();
        await page.waitForFunction(
            () => (window as Window & { __survive?: () => Survive }).__survive?.() != null,
            null,
            { timeout: 8000 },
        );
        const after = await read(page);
        assert("the survive seam is back after the reload", after !== null);

        // restored, then kept climbing from the stored value — so after >= before. A lost-state rebuild would
        // restart near 0 and fail this. The derived entity re-derived once, not doubled.
        assert(
            `runtime value survived the reload (before ${before!.n}, after ${after!.n})`,
            after!.n >= before!.n,
        );
        assert(
            `warm-derived sprout not doubled across the reload (got ${after!.sprouts})`,
            after!.sprouts === 1,
        );
        await step("restored", { highlight: [S.canvas] });
    },
);
