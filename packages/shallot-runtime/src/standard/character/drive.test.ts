import { afterEach, expect, test } from "bun:test";
import { pose, resetDrive, states, teleport } from "./drive";
import type { CharState } from "./sweep";

// A minimal registered controller state — the sweep builds this from the authored Body; here we set it
// directly (the map is the drive surface's seam) to exercise teleport without a full physics step.
function register(eid: number, pos: [number, number, number], vel: [number, number, number]): void {
    const st: CharState = {
        pos: [...pos],
        quat: [0, 0, 0, 1],
        half: 0.5,
        radius: 0.3,
        maxSlopeCos: Math.cos((45 * Math.PI) / 180),
        jumpSpeed: 0,
        vel: [...vel],
        realizedVel: [...vel],
        grounded: false,
        groundNormal: [0, 0, 0],
        coyote: 0,
        buffer: 0,
    };
    states.set(eid, st);
}

afterEach(resetDrive);

test("teleport moves the character and clears its motion", () => {
    register(1, [0, 10, 0], [0, -30, 0]); // falling fast
    expect(teleport(1, 4, 2, -1)).toBe(true);

    const out: [number, number, number] = [0, 0, 0];
    expect(pose(1, out)).toBe(true);
    expect(out).toEqual([4, 2, -1]);

    // the respawn contract: a teleported character doesn't inherit its fall velocity (else it re-dies or
    // launches the instant the next sweep integrates from the old speed).
    const st = states.get(1) as CharState;
    expect(st.vel).toEqual([0, 0, 0]);
    expect(st.realizedVel).toEqual([0, 0, 0]);
});

test("teleport of an unregistered character is a no-op returning false", () => {
    expect(teleport(99, 1, 2, 3)).toBe(false);
});
