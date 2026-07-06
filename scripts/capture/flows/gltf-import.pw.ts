import { type Page, test } from "@playwright/test";
import { defineFlow } from "../runner";
import { S } from "../selectors";

const PORT = process.env.CAPTURE_GLTF_PORT || "3006";
const BASE = `http://localhost:${PORT}`;
const SCENE = "scenes/gltf.scene";

// the editor glTF import end to end: a real DataTransfer drop on the
// viewport uploads the model through /__api/asset, loads it live, and mints an ordinary scene node — one
// undo removes the import, the serialized scene carries the by-name part node, and a fresh editor load of
// that scene re-imports it through the declarative preloader. The dropped bytes are the fixture's own
// box.gltf, so the upload takes the identical-reuse path and the fixture dir gains no residue. Runs in
// ?save=off; the round-trip writes the scene through /__api/file and restores it on the way out.

async function readScene(page: Page): Promise<string> {
    return (await page.request.get(`${BASE}/__api/file?path=${SCENE}`)).text();
}

async function writeScene(page: Page, content: string): Promise<void> {
    await page.request.post(`${BASE}/__api/file?path=${SCENE}`, { data: { content } });
}

// entities carrying a Color component, read through the control seam — the fixture's placed box is the
// only one at boot; the imported node adds a second. null while the ws is mid-(re)build.
async function colorCount(page: Page): Promise<number | null> {
    const res = await page.request.get(`${BASE}/__api/entities?component=Color`);
    if (!res.ok()) return null;
    const data = await res.json();
    return Array.isArray(data) ? data.length : null;
}

async function poll(
    page: Page,
    want: (n: number | null) => boolean,
    timeoutMs = 12_000,
): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (want(await colorCount(page))) return true;
        await page.waitForTimeout(150);
    }
    return false;
}

// build a DataTransfer from the fixture's own model files and dispatch the real drag gesture on the
// viewport container — the same events a user's OS drop fires. A synthetic DataTransfer has no
// webkitGetAsEntry, so the editor's collector takes its flat-files path.
async function dropModel(page: Page): Promise<void> {
    await page.evaluate(async () => {
        const fetchFile = async (path: string, type: string) => {
            const blob = await (await fetch(path)).blob();
            return new File([blob], path.slice(1), { type });
        };
        const dt = new DataTransfer();
        dt.items.add(await fetchFile("/box.gltf", "model/gltf+json"));
        dt.items.add(await fetchFile("/Box.bin", "application/octet-stream"));
        const target = document.querySelector(".viewport-canvas") as HTMLElement;
        const rect = target.getBoundingClientRect();
        const at = { clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 };
        const ev = (type: string) =>
            new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true, ...at });
        target.dispatchEvent(ev("dragenter"));
        target.dispatchEvent(ev("dragover"));
        // hold the enter state long enough for the overlay assert below, then drop
        await new Promise((r) => setTimeout(r, 250));
        target.dispatchEvent(ev("drop"));
    });
}

defineFlow(
    { name: "gltf-import", scene: "gltf", target: "manual", timeout: 90_000 },
    async ({ page, step, sample, assert, act, openEditor }) => {
        test.setTimeout(90_000);

        await openEditor(Number(PORT));
        const original = await readScene(page);

        try {
            const spawned = await poll(page, (n) => n === 1);
            assert("the fixture booted with its one placed box", spawned);
            const rowsBefore = await page.locator(S.row).count();

            const dropped = dropModel(page);
            // the drag-enter overlay shows while the file drag hovers (dropModel holds it ~250ms)
            await act.waitFor(".drop-overlay", { timeout: 3000 });
            await dropped;

            // the import minted an ordinary scene node: an outliner row named from the file stem
            await page.locator(S.row, { hasText: "box" }).first().waitFor({ timeout: 15_000 });
            const imported = await poll(page, (n) => n === 2);
            assert("the imported node spawned a second Color entity", imported);
            assert(
                "a new outliner row appeared",
                (await page.locator(S.row).count()) === rowsBefore + 1,
            );
            const px = await sample();
            assert("the viewport renders the scene", px.nonBackground > 0.02);
            await step("imported", { highlight: [S.outliner, S.viewport] });

            // the document round-trips the import as a plain by-name part node
            const state = (await (await page.request.get(`${BASE}/__api/state`)).json()) as {
                scene: string;
            };
            assert(
                "the serialized scene names the imported mesh",
                state.scene.includes("mesh: box.gltf#0"),
            );
            const importedScene = state.scene;

            // one undo removes the whole import
            await act.key("Control+z");
            const undone = await poll(page, (n) => n === 1, 5000);
            assert("one undo removed the imported entity", undone);
            assert("the outliner row is gone", (await page.locator(S.row).count()) === rowsBefore);
            await step("undone", { highlight: [S.outliner] });

            // round-trip: a fresh editor load of the imported scene re-imports through the declarative
            // preloader — no import code, the part node's mesh name is the load trigger. Park the page
            // first: the scene write fires the editor's own external-change reload, which would race the
            // goto below.
            await page.goto("about:blank");
            await writeScene(page, importedScene);
            await openEditor(Number(PORT));
            const reloaded = await poll(page, (n) => n === 2);
            assert("a fresh load of the saved scene shows the imported model", reloaded);
            await page.locator(S.row, { hasText: "box" }).first().waitFor({ timeout: 5000 });
            const px2 = await sample();
            assert("the reloaded scene renders", px2.nonBackground > 0.02);
            await step("round-trip", { highlight: [S.viewport] });
        } finally {
            await writeScene(page, original);
        }
    },
);
