import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, GlobalTransform, Time } from "@dylanebert/shallot";
import { local } from "@dylanebert/shallot/multiplayer";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ClientSession, GAME, scene } from "./network";

setDefaultTimeout(TIMEOUT.node);

test("a room's roster and avatar survive reconnection and track bot departure", async () => {
    const host = await local(GAME, {
        rooms: true,
        seed: 7,
        faults: { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 },
    });
    try {
        const seat = await host.join();
        const code = host.create()!;
        const privateSeat = await host.join(code);
        const a = await createApp({ defaults: false, plugins: [seat.plugin] });
        const b = await createApp({ defaults: false, plugins: [privateSeat.plugin] });
        try {
            const step = (count: number) => {
                for (let i = 0; i < count; i++) {
                    host.pump(Time.FIXED_DT);
                    a.world.step(Time.FIXED_DT);
                    b.world.step(Time.FIXED_DT);
                }
            };
            step(12);
            const peer = a.world.resource(ClientSession).client!;
            const other = b.world.resource(ClientSession).client!;
            expect(peer.roster).toBe(0b0001);
            expect(other.roster).toBe(0b0001);
            const bot = await seat.addBot({ seed: 11, input() {} });
            step(12);
            expect(peer.roster).toBe(0b0011);
            expect(other.roster).toBe(0b0001);
            const avatar = scene(a.world).players[peer.slot]!.player;
            const z = () => a.world.storage(GlobalTransform).translation.z.get(avatar);
            peer.record([0, 1, 0, 0, 0, 0]);
            step(30);
            expect(z()).toBeLessThan(11);
            seat.cut();
            peer.record([0, 0, 0, 0, 0, 0]);
            step(3);
            expect(seat.resume()).toBe(true);
            step(12);
            expect(peer.connection.generation).toBe(1);
            expect(peer.slot).toBe(0);
            expect(z()).toBeLessThan(11);
            bot.remove();
            step(12);
            expect(peer.roster).toBe(0b0001);
        } finally {
            a.dispose();
            b.dispose();
        }
        expect(host.list()).toEqual([]);
    } finally {
        host.dispose();
    }
});
