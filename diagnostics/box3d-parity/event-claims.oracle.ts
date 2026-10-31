import { expect, test } from "bun:test";
import gold from "../../src/standard/physics/world/ecs-event-claims.gold.json";
import { nativeBinary, run } from "./native";

test("contact snapshots, event records and mixed-island sleep match native observations", () => {
    expect(JSON.parse(run([nativeBinary("event-claims.c", "x86_64")]))).toEqual(gold);
});
