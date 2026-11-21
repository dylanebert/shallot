import { expect, test } from "bun:test";
import { component, entity, f16, u32, vec3, vec4, World } from "../../engine";
import { type Connection, Replicate, ReplicationReceiver, ReplicationSender, seededLink } from ".";

// 25 vec4 fields: 100 lanes per entity, about the multiplayer example's 1100 values over 11 bodies.
const Wide = component(
    "RetentionWide",
    Object.fromEntries(Array.from({ length: 25 }, (_, i) => [`f${i}`, vec4])),
);
const Link = component("RetentionLink", { target: entity, count: u32, half: f16, offset: vec3 });
const schema = [Wide, Link];
const rules = [new Map([[1, [Wide, Link]]])];

function scene(world: World) {
    const bodies: number[] = [];
    for (let i = 0; i < 11; i++) {
        const eid = world.create();
        world.add(eid, Wide);
        world.add(eid, Replicate);
        for (let f = 0; f < 25; f++) world.storage(Wide)[`f${f}`]!.set(eid, i, f, i * f, 1);
        bodies.push(eid);
    }
    return bodies;
}

function move(world: World, bodies: number[], tick: number) {
    // The last lane alternates between 0 and -0, which Object.is tells apart.
    for (const eid of bodies.slice(0, 3))
        world.storage(Wide).f0!.set(eid, tick / 7, eid, -tick, tick % 2 ? -0 : 0);
}

/** Reads the sender's private retention, as resource.test.ts reads World._resources. */
function retention(sender: ReplicationSender) {
    const { _pending, _bases } = sender as unknown as {
        _pending: Map<number, Record<string, unknown> & { base: number }>;
        _bases: Map<number, unknown>;
    };
    return { pending: _pending, bases: _bases };
}

/** No pending frame holds rows; bases are exactly those pending frames name plus the
 * acknowledged state, the next record's base. */
function expectOwnership(sender: ReplicationSender) {
    const { pending, bases } = retention(sender);
    const named = new Set<number>();
    for (const frame of pending.values()) {
        expect(Object.values(frame).some((value) => value instanceof Map)).toBe(false);
        named.add(frame.base);
    }
    for (const tick of bases.keys())
        expect(named.has(tick) || tick === sender.acknowledged).toBe(true);
    expect(bases.size).toBeLessThanOrEqual(named.size + 1);
}

function acknowledgement(tick: number) {
    const bytes = new Uint8Array(12);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 3, true);
    view.setUint32(4, tick, true);
    return bytes;
}

test("a never-acknowledging recipient's pending frames hold bytes, not captured rows", () => {
    const world = new World();
    const bodies = scene(world);
    const silent: Connection = { close() {}, send() {}, receive: () => [] };
    const sender = new ReplicationSender(world, schema, rules, 1, silent);
    for (let tick = 0; tick < 600; tick++) {
        move(world, bodies, tick);
        sender.record(tick);
        sender.flush();
    }
    expect(retention(sender).pending.size).toBe(600);
    expectOwnership(sender);
    world.dispose();
});

test("a draining backlog keeps only the bases its pending frames name and the acknowledged state", () => {
    const world = new World();
    const bodies = scene(world);
    const inbox: Uint8Array[] = [];
    const connection: Connection = { close() {}, send() {}, receive: () => inbox.splice(0) };
    const sender = new ReplicationSender(world, schema, rules, 1, connection);
    // Every frame recorded before an acknowledgement names the image as its base.
    for (let tick = 0; tick <= 40; tick++) {
        move(world, bodies, tick);
        sender.record(tick);
        sender.flush();
    }
    for (let tick = 0; tick < 40; tick++) {
        inbox.push(acknowledgement(tick));
        sender.flush();
        expect(sender.acknowledged).toBe(tick);
        expectOwnership(sender);
    }
    expect(retention(sender).pending.size).toBe(1);
    world.dispose();
});

/** Every replicated row of a World, independent of eid space and query order. A Link names its
 * target by the target's first Wide field. -0 stays distinct from 0. */
function replicatedState(world: World): string[] {
    const text = (value: unknown) =>
        JSON.stringify(value, (_key, v) => (Object.is(v, -0) ? "-0" : v));
    const wide = (eid: number) => {
        const row = world.get(eid, Wide);
        return row ? text(Object.values(row)) : "none";
    };
    const rows: string[] = [];
    for (const eid of world.query([Wide])) rows.push(`wide ${wide(eid)}`);
    for (const eid of world.query([Link])) {
        const link = world.get(eid, Link)!;
        const target = link.target ? wide(link.target) : "null";
        rows.push(`link ${text([link.count, link.half, link.offset])} ${target}`);
    }
    return rows.sort();
}

test("each applied frame equals the server's replicated state through loss, retransmission and a stale acknowledgement", () => {
    const server = new World();
    const client = new World();
    const link = seededLink(0x7e7d, { minDelay: 1, maxDelay: 9, loss: 0.2, duplication: 0.2 });
    let sends = 0;
    let clock = 0;
    const send = link.a.send.bind(link.a);
    link.a.send = (bytes) => {
        sends++;
        send(bytes);
    };
    let stale: Uint8Array | undefined;
    const sendAck = link.b.send.bind(link.b);
    link.b.send = (bytes) => {
        if (clock === 30) stale = bytes.slice();
        sendAck(bytes);
    };
    const receive = link.a.receive.bind(link.a);
    link.a.receive = () => {
        const messages = receive();
        // An acknowledgement of an older frame arrives after newer ones were acknowledged.
        if (clock === 90 && stale) messages.push(stale);
        return messages;
    };
    const bodies = scene(server);
    const owner = server.create();
    server.add(owner, Replicate);
    server.add(owner, Link, { target: bodies[0]!, count: 0, half: 0.5, offset: [-0, 1, 2] });
    const sender = new ReplicationSender(server, schema, rules, 1, link.a);
    const confirmed: number[] = [];
    const expected = new Map<number, string[]>();
    const receiver = new ReplicationReceiver(client, schema, link.b, (tick) => {
        confirmed.push(tick);
        expect(replicatedState(client)).toEqual(expected.get(tick)!);
    });
    try {
        for (; clock < 160; clock++) {
            link.advance(clock);
            move(server, bodies, clock);
            if (server.has(owner, Link)) {
                server.storage(Link).half.set(owner, clock / 3);
                server.storage(Link).offset.set(owner, clock % 2 ? 0 : -0, clock, -clock);
            }
            if (clock === 20) server.destroy(bodies.pop()!);
            if (clock === 35) server.remove(owner, Link);
            if (clock === 50) server.add(owner, Link, { target: bodies[1]!, count: clock });
            if (clock === 70) bodies.push(scene(server)[0]!);
            server.tick();
            expected.set(clock, replicatedState(server));
            sender.record(clock);
            sender.flush();
            expectOwnership(sender);
            // The receiver falls silent for 40 ticks, so frames pend and time out.
            if (clock < 55 || clock >= 95) receiver.flush();
        }
        for (; clock < 400; clock++) {
            link.advance(clock);
            sender.flush(clock);
            expectOwnership(sender);
            receiver.flush();
        }
        expect(confirmed).toEqual(Array.from({ length: 160 }, (_, i) => i));
        expect(stale).toBeDefined();
        expect(link.counts.dropped).toBeGreaterThan(0);
        expect(sends).toBeGreaterThan(160);
    } finally {
        server.dispose();
        client.dispose();
    }
});
