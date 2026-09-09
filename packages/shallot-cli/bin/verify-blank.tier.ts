import { expect, test } from "bun:test";
import { verify } from "../../../scripts/verify";
import { requireDisplay } from "../../shallot/tests/flows/harness";

// By-path headed tier for verify's own pixel-honest `rendered` verdict — the red proof, moved here from
// `examples/flows/blank` because the claim is about this CLI's gate, not about examples. The fixture app
// (`packages/shallot/tests/flows/blank`, beside the other relocated flows) draws nothing and its harness
// reports ok:true, and it deliberately does NOT declare `noRender`. So verify MUST fail it with
// `rendered: false`. This tier passes when verify correctly goes red — an expected-fail, which is what
// keeps a standing proof of the gate from turning the matrix red on every run.
//
//     bun test ./packages/shallot-cli/bin/verify-blank.tier.ts
//
// `verdict.ok` is asserted too: verify's harness-ready-timeout path returns the same
// `pass:false`/`rendered:false` shape, so a bitrotted fixture whose harness never installs would
// vacuously satisfy the red proof. Asserting the fixture's own verdict succeeded pins the failure to the
// pixel gate reading the blank, not to a broken boot.

test("verify reds a canvas that drew nothing (rendered:false, harness passing)", async () => {
    requireDisplay();
    const result = await verify("packages/shallot/tests/flows/blank", ["--timeout", "60000"]);
    expect({
        pass: result?.pass,
        rendered: result?.rendered,
        verdict: result?.verdict?.ok,
    }).toEqual({ pass: false, rendered: false, verdict: true });
}, 180_000);
