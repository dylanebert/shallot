import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { PlayerMotion } from "../../../examples/first-person/src/motion";
import { Player, PlayerInput, PlayerPlugin } from "../../../examples/first-person/src/player";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { Character } from "../../standard/physics";
import { IntentReceiver, IntentSender, seededLink } from ".";

setDefaultTimeout(TIMEOUT.node);
test("a press whose first message is dropped survives as one accepted operation and launches first-person's built-in player", async () => {
    const app = await createApp({ defaults: false, plugins: [PlayerPlugin] });
    const world = app.world;
    try {
        const floor = world.create();
        world.add(floor, Transform);
        world.add(floor, Body, { type: BodyType.Static });
        world.add(floor, Shape, { scale: [40, 1, 40, 0] });
        const player = world.create();
        world.add(player, Transform, { translation: [0, 1.5, 0] });
        world.add(player, Body, { type: BodyType.Kinematic });
        world.add(player, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, -0.6, 0],
            capsuleCenter2: [0, 0.6, 0],
            radius: 0.3,
        });
        world.add(player, Character);
        world.add(player, Player);
        world.add(player, PlayerInput);
        for (let i = 0; i < 60; i++) world.tick();
        const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
        let dropped = 0;
        const sender = new IntentSender({
            close: link.b.close,
            receive: link.b.receive,
            send(bytes) {
                if (dropped++ === 0) return;
                link.b.send(bytes);
            },
        });
        const receiver = new IntentReceiver(link.a);
        sender.record(61, "operation", [1]);
        sender.flush();
        receiver.receive(link.a.receive(), 61);
        expect(receiver.records.size).toBe(0);
        world.tick();
        sender.flush();
        receiver.receive(link.a.receive(), 62);
        sender.flush(link.b.receive());
        const accepted = [...receiver.records.values()];
        expect(accepted).toHaveLength(1);
        expect(accepted[0]!.tick).toBe(62);
        expect(accepted[0]!.lateness).toBe(1);
        const before = world.snapshot();
        const launch = () => {
            world
                .storage(PlayerInput)
                .jumpPresses.set(
                    player,
                    accepted.filter((r) => r.tick <= world.time.fixedTick + 1).length,
                );
            world.tick();
            return [
                world.storage(PlayerMotion).lastJumpPresses.get(player),
                world.storage(Character).velocity.y.get(player),
            ];
        };
        const first = launch();
        world.restore(before);
        const replay = launch();
        expect(first).toEqual(replay);
        expect(first[0]).toBe(1);
        expect(first[1]).toBeGreaterThan(8);
        let highest = 0;
        for (let i = 0; i < 30; i++) {
            world.tick();
            highest = Math.max(highest, world.storage(GlobalTransform).translation.y.get(player));
        }
        expect(highest).toBeGreaterThan(2.5);
        sender.flush();
        receiver.receive(link.a.receive(), 100);
        expect(receiver.records.size).toBe(1);
    } finally {
        app.dispose();
    }
});
