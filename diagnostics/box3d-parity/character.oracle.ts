import { expect, test } from "bun:test";
import { characterGold, visitCharacterGold } from "../../src/standard/physics/character-kernel.fixture";
import { nativeBinary, run } from "./native";

test("the kernel sample mover matches regenerated native Box3D movement records", async () => {
    const native = JSON.parse(run([nativeBinary("character.c")])) as typeof characterGold;
    expect(native).toEqual(characterGold);
    await visitCharacterGold(native, (actual, expected, name) => expect(actual, name).toEqual(expected));
    console.log(JSON.stringify({cases:native.length, events:native.reduce((n,row) => n+row.events.length,0)}));
});
