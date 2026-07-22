import { Body, Physics, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

// dynamics smoke for `shallot verify` (run headless by `scripts/recipes.ts`): assert the car advances under
// throttle — the concept drive-a-vehicle teaches. It clicks the canvas (focus), holds W (the throttle key
// `Inputs` listens for on the window), and checks the car travelled. Kept out of the teaching plugin
// (car.ts); enabled only through this recipe's manifest.

const SETTLE_MS = 800; // let the car drop onto its wheels before throttling
const SAMPLE_MS = 2400;

const frame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));
const dynamic = (state: State): number[] =>
    [...state.query([Body])].filter((e) => Body.mass.get(e) > 0);

async function wait(ms: number): Promise<void> {
    const t0 = performance.now();
    while (performance.now() - t0 < ms) await frame();
}

// mirror a real user's click on the canvas before pressing a key — defensive only, not required for key
// delivery: `canvasFocused` defaults true (input/index.ts keyDown reads it, but a fresh canvas already
// passes), so headless verify's missing click does not drop keydowns. a pointerdown sets the focus flag; the
// window pointerup releases the (unused) left button so nothing stays held.
function focusCanvas(): void {
    const canvas = document.querySelector("canvas");
    if (!canvas) return;
    canvas.dispatchEvent(
        new PointerEvent("pointerdown", {
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 8,
            clientY: 8,
            bubbles: true,
        }),
    );
    window.dispatchEvent(
        new PointerEvent("pointerup", { pointerId: 1, button: 0, buttons: 0, bubbles: true }),
    );
}

export const Smoke: Plugin = {
    name: "CarSmoke",
    warm(state: State) {
        const h = installHarness(state);
        h.run = async (): Promise<Verdict> => {
            await wait(SETTLE_MS);
            const eids = dynamic(state);
            const sx = new Map<number, number>();
            const sz = new Map<number, number>();
            for (const e of eids) {
                const b = Physics.backend?.readBody(e);
                if (b) {
                    sx.set(e, b.pos[0]);
                    sz.set(e, b.pos[2]);
                }
            }
            // click the canvas to focus it, then hold the throttle
            focusCanvas();
            window.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyW" }));
            await wait(SAMPLE_MS);
            window.dispatchEvent(new KeyboardEvent("keyup", { code: "KeyW" }));
            let advance = 0;
            for (const e of eids) {
                const b = Physics.backend?.readBody(e);
                if (b) {
                    advance = Math.max(
                        advance,
                        Math.hypot(b.pos[0] - (sx.get(e) ?? 0), b.pos[2] - (sz.get(e) ?? 0)),
                    );
                }
            }
            const ok = advance > 1;
            return {
                ok,
                checks: [
                    {
                        name: "car advances under throttle",
                        ok,
                        detail: `${advance.toFixed(2)} m travelled while W held (need > 1)`,
                    },
                ],
            };
        };
    },
};

export default Smoke;
