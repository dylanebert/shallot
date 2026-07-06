import { type Page, test } from "@playwright/test";
import { defineFlow } from "../runner";
import { S } from "../selectors";

const PORT = process.env.CAPTURE_GLTF_PORT || "3006";
const BASE = `http://localhost:${PORT}`;
const MODEL = "public/box.gltf";

// the live glTF asset-swap (roadmap glTF sub-stage 5). The path mapping + targeted invalidation are bun-
// tested (project/vite.test.ts assetSrc, gltf/decode.test.ts); this drives the wiring those can't reach —
// the dev server's public-dir watcher → `shallot:asset` → the editor's `invalidate(src)` + rebuild — against
// the real editor. Re-saving the model on disk re-decodes it off-thread and re-renders WITHOUT a page reload:
// a window sentinel survives (a reload would wipe it) and the box's Color tracks the new baseColorFactor.
// Runs in ?save=off (the editor never writes the model; the model write goes through /__api/file directly,
// which the watcher sees as external), and the model is restored on the way out.

async function readModel(page: Page): Promise<string> {
    return (await page.request.get(`${BASE}/__api/file?path=${MODEL}`)).text();
}

async function writeModel(page: Page, content: string): Promise<void> {
    await page.request.post(`${BASE}/__api/file?path=${MODEL}`, { data: { content } });
}

// the strongest green channel across the scene's Color entities, read through the control seam — the box is
// the only thing that turns green (editor chrome uses the gold accent, not green), so this isolates it
// without depending on entity order. null while the ws is mid-rebuild.
async function boxGreen(page: Page): Promise<number | null> {
    const res = await page.request.get(`${BASE}/__api/entities?component=Color`);
    if (!res.ok()) return null;
    // mid-rebuild the seam answers `{ error: "No engine running" }` (ecs null), not the entity array — treat
    // that as "not ready yet" and let the caller keep polling
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) return null;
    // components key by their registered (lowercase scene) name, so the Color component reads as `color`
    const entities = data as { components: Record<string, Record<string, number>> }[];
    return Math.max(...entities.map((e) => e.components.color?.["rgba.y"] ?? 0));
}

async function poll(
    page: Page,
    want: (g: number | null) => boolean,
    timeoutMs = 12_000,
): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (want(await boxGreen(page))) return true;
        await page.waitForTimeout(150);
    }
    return false;
}

defineFlow(
    { name: "asset-swap", scene: "gltf", target: "manual", timeout: 60_000 },
    async ({ page, step, sample, assert, openEditor }) => {
        test.setTimeout(60_000);

        await openEditor(Number(PORT));

        const original = await readModel(page);
        assert("the box model loaded from disk", original.includes("baseColorFactor"));

        // the importer spawned the box and gave it the model's red baseColorFactor ([0.8, 0, 0, 1])
        const spawned = await poll(page, (g) => g !== null && g < 0.1);
        assert("the glTF box spawned with its red baseColor (low green)", spawned);
        await step("loaded", { highlight: [S.viewport] });

        // a sentinel a page reload would wipe — proves the swap is an in-place rebuild, not a reload
        await page.evaluate(() => {
            (window as unknown as { __swapMark?: string }).__swapMark = "kept";
        });

        try {
            // re-save the model with a green baseColorFactor — the watcher signals `shallot:asset`, the
            // editor invalidates "box.gltf" + rebuilds, which re-decodes off-thread and re-spawns the box
            await writeModel(
                page,
                original.replace(
                    /"baseColorFactor":\s*\[[^\]]*\]/,
                    '"baseColorFactor": [0, 0.8, 0, 1]',
                ),
            );

            const swapped = await poll(page, (g) => g !== null && g > 0.5);
            assert("re-saving the model live re-decoded + re-rendered the box green", swapped);

            const mark = await page.evaluate(
                () => (window as unknown as { __swapMark?: string }).__swapMark,
            );
            assert(
                "the swap was an in-place rebuild, not a page reload (sentinel survived)",
                mark === "kept",
            );

            const px = await sample();
            assert("the re-rendered scene is non-blank", px.nonBackground > 0.02);
            await step("swapped", { highlight: [S.viewport] });
        } finally {
            await writeModel(page, original);
        }
    },
);
