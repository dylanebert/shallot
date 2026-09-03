import { type Plugin, Sprite, SpriteFill, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/** Real-page smoke: the radial sprite meter advances its fill fraction. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const meter =
                [...state.query([Sprite])].find(
                    (eid) => Sprite.fillMode.get(eid) === SpriteFill.Radial,
                ) ?? -1;
            const before = meter < 0 ? 0 : Sprite.fill.get(meter);
            for (let i = 0; i < 30; i++) await frame();
            const after = meter < 0 ? 0 : Sprite.fill.get(meter);
            const ok = meter >= 0 && Math.abs(after - before) > 0.01;
            return {
                ok,
                checks: [
                    {
                        name: "radial sprite meter advances",
                        ok,
                        detail: `fill ${before.toFixed(3)} -> ${after.toFixed(3)}`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
