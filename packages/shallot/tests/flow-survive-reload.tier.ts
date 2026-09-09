import { expect, test } from "bun:test";
import { verify } from "../../../scripts/verify";
import { requireDisplay } from "./flows/harness";

// By-path headed tier for serialize/restore across a real page reload, moved here from
// `examples/flows/survive-reload` because the claim is about this package's state lifecycle. The fixture
// under `flows/survive-reload` self-drives the reload; verify's unified wait polls across the
// self-navigation by construction, and the restored boot's harness reports both arms.
//
//     bun test ./packages/shallot/tests/flow-survive-reload.tier.ts
//
// Both named checks are asserted, never just `verdict.ok`: a harness that readies without a run()
// reports ok:true as a bare boot smoke, and the assertions are the point.

const CHECKS = ["runtime value survived the reload", "warm-derived sprout not doubled"];

test("survive-reload: a runtime value and the warm-derived set survive a real reload", async () => {
    requireDisplay();
    const result = await verify("packages/shallot/tests/flows/survive-reload", [
        "--timeout",
        "60000",
    ]);
    expect(result?.pass).toBe(true);
    expect(result?.verdict?.ok).toBe(true);
    for (const name of CHECKS)
        expect(result?.verdict?.checks?.find((check) => check.name === name)?.ok).toBe(true);
}, 180_000);
