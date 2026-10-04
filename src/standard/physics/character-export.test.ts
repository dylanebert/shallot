import { expect, test } from "bun:test";
import { Character as RootCharacter, CharacterPlugin as RootPlugin } from "@dylanebert/shallot";
import {
    Character,
    CharacterPlugin,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";

test("the root character is standard physics's velocity-driven character", () => {
    expect(RootCharacter).toBe(Character);
    expect(RootPlugin).toBe(CharacterPlugin);
    expect(CharacterPlugin.dependencies).toContain(StandardPhysicsPlugin);
});
