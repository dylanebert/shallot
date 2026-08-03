import { type Plugin, type State, Text, text } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";
import { Charge, Readout } from "./reduce";

const READY_MS = 5000;
const EXPECTED = "total charge: 6.00";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

export const Smoke: Plugin = {
    name: "ComputeReadbackSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const expected = text(EXPECTED);
            const charges = [...state.query([Charge])]
                .map((eid) => Charge.amount.get(eid))
                .sort((a, b) => a - b);
            const authored = charges.length === 3 && charges.every((value, i) => value === i + 1);
            let content = 0;
            const start = performance.now();
            while (performance.now() - start < READY_MS) {
                content = 0;
                for (const eid of state.query([Readout, Text])) {
                    content = Text.content.get(eid);
                    break;
                }
                if (content === expected) break;
                await frame();
            }
            const ok = authored && content === expected;
            return {
                ok,
                checks: [
                    {
                        name: "three charges reduce to 6.00",
                        ok,
                        detail: `charges [${charges.join(", ")}], Text.content ${content} (expected ${expected} for "${EXPECTED}")`,
                    },
                ],
            };
        };
    },
};

export default Smoke;
