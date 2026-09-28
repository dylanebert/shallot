import { expect, test } from "bun:test";
import { build, CharacterPlugin, InputPlugin, PhysicsPlugin } from "@dylanebert/shallot";
import { Demo } from "./demo";

test("the first-person app refuses without WebGPU and names the acquisition cause", async () => {
    const message = await build({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo],
    }).then(
        (app) => {
            app.dispose();
            return "build unexpectedly succeeded";
        },
        (error: unknown) => (error instanceof Error ? error.message : String(error)),
    );
    expect(message).toContain("WebGPU not supported in this browser");
});
