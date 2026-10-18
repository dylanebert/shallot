import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { characterGold, visitCharacterGold } from "./character-kernel.fixture";

setDefaultTimeout(TIMEOUT.node);

test("Character publishes the native sample mover's full movement record", async () => {
    await visitCharacterGold(characterGold, (actual, expected, name) =>
        expect(actual, name).toEqual(expected),
    );
});
