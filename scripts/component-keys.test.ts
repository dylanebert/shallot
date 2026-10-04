import { expect, test } from "bun:test";
import { componentKeys } from "./component-keys";
import baseline from "./component-keys.base.json";

// Captured from e4acecbbe5a0b742cc6fa238b913f69724f7f3b9, not regenerated from declarations.
test("default, extra and example compositions preserve base registered keys and reflected fields", async () => {
    expect(await componentKeys()).toEqual(baseline);
});
