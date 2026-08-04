import { expect, test } from "bun:test";
import { State } from "@dylanebert/shallot";
import { ownTagHookResource } from "./render";

function resource(): { destroys: number; destroy(): void } {
    return {
        destroys: 0,
        destroy() {
            this.destroys++;
        },
    };
}

test("the tag-hook producer resource follows direct State disposal", () => {
    const state = new State();
    const owned = ownTagHookResource(state, resource());

    state.dispose();

    expect(owned.destroys).toBe(1);
});

test("an old tag-hook State cannot destroy its replacement generation", () => {
    const oldState = new State();
    const replacementState = new State();
    const old = ownTagHookResource(oldState, resource());
    const replacement = ownTagHookResource(replacementState, resource());

    oldState.dispose();
    expect(old.destroys).toBe(1);
    expect(replacement.destroys).toBe(0);

    replacementState.dispose();
    expect(replacement.destroys).toBe(1);
});
