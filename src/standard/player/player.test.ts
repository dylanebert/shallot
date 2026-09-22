import {
    Body,
    build,
    Camera,
    Character,
    CharacterPlugin,
    devices,
    InputPlugin,
    PhysicsPlugin,
    Player,
    PlayerControlSystem,
    pointerLockChanged,
    pointerMove,
    pressKey,
    readBody,
    releaseKey,
    ShapeKind,
    Time,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";

check(
    "Player control consumes supplied facts on the CPU path",
    {
        claim: "the public Player controller consumes held, released and neutral input to look and drive an actual Character without a renderer or browser input",
    },
    async () => {
        const app = await build({
            defaults: false,
            plugins: [InputPlugin, CharacterPlugin, PhysicsPlugin, TransformsPlugin],
            setup: (state) => state.addSystem(PlayerControlSystem, "Player"),
        });
        try {
            const state = app.state;
            const floor = state.create();
            state.add(floor, Body);
            Body.shape.set(floor, ShapeKind.Box);
            Body.pos.set(floor, 0, 0, 0, 0);
            Body.halfExtents.set(floor, 4, 0.5, 4, 0);
            Body.mass.set(floor, 0);

            const camera = state.create();
            state.add(camera, Camera);
            state.add(camera, Transform);

            const player = state.create();
            state.add(player, Body);
            state.add(player, Character);
            state.add(player, Player);
            Body.shape.set(player, ShapeKind.Capsule);
            Body.pos.set(player, 0, 1.3, 0, 0);
            Body.halfExtents.set(player, 0, 0.5, 0, 0.3);
            Body.mass.set(player, 0);
            Player.speed.set(player, 6);
            Player.sprint.set(player, 1);
            Player.sensitivity.set(player, 1.5);
            Player.camera.set(player, camera);
            Character.jumpSpeed.set(player, 7);
            Character.gravity.set(player, -30);

            // Establish the floor contact before the supplied jump edge arrives.
            state.step(Time.FIXED_DT);
            const initial = readBody(state, player);
            if (!initial) throw new Error("Player body did not enter the CPU physics world");
            const initialYaw = Player.yaw.get(player);
            const initialPitch = Player.pitch.get(player);

            pointerLockChanged(state, true);
            pointerMove(state, 0, 0, 12, -4);
            pressKey(state, "KeyW");
            pressKey(state, "Space");
            state.step(Time.FIXED_DT);
            const lookScale = Player.sensitivity.get(player) / 1080;
            if (Math.abs(Player.yaw.get(player) - (initialYaw - 12 * lookScale)) > 0.000001)
                throw new Error("Player did not consume the supplied locked look sensitivity");
            if (Math.abs(Player.pitch.get(player) - (initialPitch + 4 * lookScale)) > 0.000001)
                throw new Error("Player did not consume the supplied vertical look sensitivity");
            if (!devices(state).keys.held.has("KeyW"))
                throw new Error("Player lost the held move fact");

            // PlayerControlSystem writes the intent in simulation; the next fixed tick is the real
            // Character consumer. This deliberately uses the stepped clock rather than a private drive.
            state.step(Time.FIXED_DT);
            const moved = readBody(state, player);
            if (
                !moved ||
                Math.hypot(moved.pos[0] - initial.pos[0], moved.pos[2] - initial.pos[2]) < 0.001
            )
                throw new Error("Character did not apply Player's supplied movement intent");
            if (!moved || moved.pos[1] <= initial.pos[1] + 0.01)
                throw new Error("Character did not apply Player's supplied jump edge");

            releaseKey(state, "KeyW");
            state.step(Time.FIXED_DT); // the released fact reaches PlayerControlSystem
            const beforeNeutral = readBody(state, player);
            if (!beforeNeutral) throw new Error("Player body disappeared after release");
            state.step(Time.FIXED_DT); // the first neutral frame drains the previous simulation intent
            const neutral = readBody(state, player);
            if (!neutral) throw new Error("Player body disappeared on the neutral step");
            state.step(Time.FIXED_DT); // this fixed tick must not replay a stale movement intent
            const settled = readBody(state, player);
            if (!settled) throw new Error("Player body disappeared on the settled neutral step");
            if (
                Math.hypot(settled.pos[0] - neutral.pos[0], settled.pos[2] - neutral.pos[2]) >
                0.0001
            )
                throw new Error("released Player movement was replayed after the neutral step");
        } finally {
            app.dispose();
        }
    },
);
