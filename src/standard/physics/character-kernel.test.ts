import { expect, test } from "bun:test";
import { characterGold, visitCharacterGold } from "./character-kernel.fixture";

test("Character publishes the native sample mover's full movement record", async () => {
    await visitCharacterGold(characterGold, (actual, expected, name) =>
        expect(actual, name).toEqual(expected),
    );
});
