import { expect, test } from "bun:test";
import {
    Character as TransitionalCharacter,
    CharacterPlugin as TransitionalPlugin,
} from "@dylanebert/shallot";
import {
    Character as CharacterSubpath,
    CharacterPlugin as TransitionalSubpath,
} from "@dylanebert/shallot/character";
import {
    Character,
    CharacterPlugin,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";

test("the velocity-driven character is published only by standard physics while the root keeps the transitional character", () => {
    expect(Character).toBeDefined();
    expect(CharacterPlugin.dependencies).toContain(StandardPhysicsPlugin);
    expect(Character).not.toBe(TransitionalCharacter);
    expect(CharacterPlugin).not.toBe(TransitionalPlugin);
    expect(TransitionalCharacter).toBe(CharacterSubpath);
    expect(TransitionalPlugin).toBe(TransitionalSubpath);
});
