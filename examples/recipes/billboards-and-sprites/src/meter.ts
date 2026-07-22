import { Sprite, SpriteFill, type State, type System } from "@dylanebert/shallot";

// `fillMode: radial` shows only the leading `fill` fraction of a sprite's image, sweeping clockwise
// from the top, so a ring reads as a progress meter. `vertical`/`horizontal` fill tanks and bars the
// same way.
export const meter = {
    name: "meter",
    group: "simulation",
    update(state: State) {
        const t = (state.time.elapsed % 3) / 3; // loop 0→1 every three seconds
        for (const eid of state.query([Sprite]))
            if (Sprite.fillMode.get(eid) === SpriteFill.Radial) Sprite.fill.set(eid, t);
    },
} satisfies System;
