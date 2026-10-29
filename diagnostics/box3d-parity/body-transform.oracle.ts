import { expect, setDefaultTimeout, test } from "bun:test";
import { nativeBinary, run } from "./native";

setDefaultTimeout(0);
test("body transform gold is native b3Body_GetTransform", async () => {
    expect(run([nativeBinary("body-transform.c")])).toBe(
        await Bun.file(new URL("../../src/standard/physics/body-transform.gold.txt", import.meta.url)).text(),
    );
});
