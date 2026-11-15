import { Body, Shape } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { World } from "../../engine";
import { Replicate, ReplicationReceiver, ReplicationSender, seededLink } from ".";

const server = new World();
const client = new World();
const schema = [Transform, GlobalTransform, Body, Shape];
try {
    for (let i = 0; i < 10; i++) {
        const eid = server.create();
        server.add(eid, Replicate);
        for (const component of schema) server.add(eid, component);
    }
    const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    const sender = new ReplicationSender(server, schema, [new Map([[1, schema]])], 1, link.a);
    const receiver = new ReplicationReceiver(client, schema, link.b);
    function measure(tick: number, label: string) {
        sender.record(tick);
        sender.flush();
        const packets = link.b.receive();
        console.log(
            `${label}: ${packets.reduce((total, bytes) => total + bytes.length, 0)} bytes for 10 bodies`,
        );
        for (const packet of packets) link.a.send(packet);
        receiver.flush();
        sender.flush();
    }
    measure(0, "image");
    measure(1, "unchanged frame");
    for (const eid of server.query([Transform])) {
        server.storage(Transform).translation.set(eid, 1, 2, 3);
        server.storage(GlobalTransform).translation.set(eid, 1, 2, 3);
    }
    measure(2, "translated frame");
} finally {
    server.dispose();
    client.dispose();
}
