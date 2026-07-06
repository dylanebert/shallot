import { defineFlow } from "../runner";
import { S } from "../selectors";

// the zoo editor-capture validation sweep (testing.md "Editor tiers", roadmap "Project shape +
// onboarding"). Every zoo specimen (`examples/zoo/<module>/`) is a real, untouched create-shallot
// project — its own `shallot.json` config + scene + public dir. capture.ts boots one editor server per
// specimen and passes the list as CAPTURE_ZOO; this one flow drives them all in a single browser
// session (`target: "manual"`, navigating each in turn) so the browser launch + WSL→Windows staging
// is paid once. Two things ride the same pass: the gate (each specimen opens in the editor, surfaces no
// error, and renders non-background pixels — the editor tier's render contract) and the docs artifact
// (one screenshot per specimen, keyed by module name, feeding the UI tabs).

interface Specimen {
    name: string;
    port: number;
}

const SPECIMENS: Specimen[] = JSON.parse(process.env.CAPTURE_ZOO || "[]");

// a specimen may import async content before it renders — sponza fetches + decodes a ~9MB glTF (Draco
// geometry + KTX2 textures), well past the editor's canvas-ready settle. So poll each specimen for its
// first rendered frame, bounded, rather than assuming the post-canvas settle covers its load.
const RENDER_TIMEOUT = 15_000;
const POLL_MS = 500;
const RENDERED = 0.1; // non-background fraction that counts as "the scene is on screen"

// one test drives every specimen, so the budget scales with the count (15s base fits one page)
const FLOW_TIMEOUT = 15_000 + SPECIMENS.length * (RENDER_TIMEOUT + 8_000);

defineFlow(
    { name: "zoo-sweep", scene: "zoo", target: "manual", timeout: FLOW_TIMEOUT },
    async ({ openEditor, sample, step, assert, act, page }) => {
        assert("at least one zoo specimen discovered", SPECIMENS.length > 0);

        // the editor surfaces an authoring problem in its DOM, not as a thrown error: a build failure as a
        // `.banner`, and a `diagnose(doc)` finding (e.g. a scene attribute for an unregistered component —
        // an unresolved plugin) as the Issues badge, its messages in the summoned popover. A specimen that
        // renders empty does so *because* of one of these, so read them and fail with the cause, not a bare
        // "non-background 0.01".
        const editorErrors = async (): Promise<string[]> => {
            const out: string[] = [];
            for (const t of await page.$$eval(`${S.banner} .text`, (els) =>
                els.map((e) => e.textContent?.trim() ?? ""),
            ))
                if (t) out.push(`banner: ${t}`);
            // the Issues badge is in the DOM whenever diagnose(doc) is non-empty; the messages live in the
            // summoned popover, so open it to read each diagnostic.
            if (await page.$(".issues .badge")) {
                await page.click(".issues .badge");
                for (const m of await page.$$eval(".issues .popover .row .message", (els) =>
                    els.map((e) => e.textContent?.trim() ?? ""),
                ))
                    if (m) out.push(`issue: ${m}`);
            }
            return out;
        };

        for (const { name, port } of SPECIMENS) {
            await openEditor(port);
            // wait on the render condition, not a fixed delay — orbit is instant, sponza decodes for seconds
            let shot = await sample();
            for (let t = 0; shot.nonBackground <= RENDERED && t < RENDER_TIMEOUT; t += POLL_MS) {
                await act.wait(POLL_MS);
                shot = await sample();
            }
            const errors = await editorErrors();
            const detail = errors.length
                ? errors.join("; ")
                : `non-background ${shot.nonBackground.toFixed(2)}`;
            assert(
                `${name}: opens in the editor and renders (${detail})`,
                errors.length === 0 && shot.nonBackground > RENDERED,
            );
            await step(name, { highlight: [S.viewport] });
        }
    },
);
