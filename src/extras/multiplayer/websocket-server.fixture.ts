import { createInterface } from "node:readline";
import { component, u32, World } from "../../engine";
import { IntentReceiver, Replicate, ReplicationSender } from ".";
import { type ServerWebSocketConnection, serveWebSocket } from "./bun";

const Value = component("SocketValue", { value: u32 });
const world = new World();
let tick = 10;
let next = 0;
let closed = 0;
const peers = new Map<
    ServerWebSocketConnection,
    {
        id: number;
        eid: number;
        sender: ReplicationSender;
        intent: IntentReceiver;
    }
>();
const server = serveWebSocket({
    hostname: "127.0.0.1",
    port: 0,
    connect(connection) {
        const id = ++next;
        const eid = world.create();
        world.add(eid, Value, { value: id * 100 });
        world.add(eid, Replicate, { rule: id });
        const rules = Array.from(
            { length: id + 1 },
            (_, rule) => new Map(rule === id ? [[id, [Value]]] : []),
        );
        const sender = new ReplicationSender(world, [Value], rules, id, connection);
        peers.set(connection, { id, eid, sender, intent: new IntentReceiver(connection) });
        sender.record(tick);
        sender.flush();
    },
    disconnect(connection) {
        world.destroy(peers.get(connection)!.eid);
        peers.delete(connection);
        closed++;
    },
});
console.log(JSON.stringify({ port: server.port }));
for await (const line of createInterface({ input: process.stdin })) {
    const command = JSON.parse(line) as { step?: boolean };
    for (const peer of peers.values()) peer.intent.receive(peer.sender.flush(), tick + 1);
    if (command.step) {
        tick++;
        for (const peer of peers.values()) {
            const input = peer.intent.inputAt(tick);
            if (input) world.storage(Value).value.set(peer.eid, input.values[0]!);
            peer.sender.record(tick, [...peer.intent.records.values()]);
            peer.sender.flush();
        }
    }
    console.log(
        JSON.stringify({
            tick,
            closed,
            peers: [...peers.values()].map((peer) => ({
                id: peer.id,
                value: world.get(peer.eid, Value)!.value,
                inputs: peer.intent.inputs.size,
            })),
        }),
    );
}
server.stop(true);
world.dispose();
