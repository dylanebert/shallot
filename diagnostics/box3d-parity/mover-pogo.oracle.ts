import { expect, test } from "bun:test";
import gold from "../../crates/physics/tests/mover_pogo.gold.json";
import { visitMoverPogo, type MoverPogoCase } from "../../src/standard/physics/solver/mover-pogo.fixture";
import { nativeBinary, run } from "./native";

const native = JSON.parse(run([nativeBinary("mover-pogo.c")])) as MoverPogoCase[];
test("the recorded mover and pogo cases equal regenerated native records", () => expect(native).toEqual(gold));
for (const c of native) {
    const kind=c.kind===9?"mover":"pogo";
    const mode=["compression and force saturation","tension and force saturation","recreated warm-started state"][c.mode];
    test(`${kind}: ${mode} equals native Box3D bit-for-bit`, () => {
        visitMoverPogo(c,(actual,expected,label)=>expect(actual,label).toEqual(expected));
    });
}
