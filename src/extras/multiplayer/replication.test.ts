import { expect, test } from "bun:test";
import { component, entity, f16, f32, i32, u8, u16, u32, vec3, World } from "../../engine";
import {
    Replicate,
    type ReplicatedRows,
    ReplicationReceiver,
    ReplicationSender,
    seededLink,
} from ".";

const Position = component("NetworkTestPosition", { value: vec3, count: u32 });
const Target = component("NetworkTestTarget", { target: entity });
const Secret = component("NetworkTestSecret", { value: f32 });
const schema = [Position, Target, Secret];

function exercise(
    join: number,
    check: (
        server: World,
        client: World,
        receiver: ReplicationReceiver,
        rows: ReplicatedRows,
        expected: Map<number, number[]>,
        tick: number,
        target: number[] | null | undefined,
    ) => void,
) {
    const server = new World();
    const client = new World();
    // Different eid spaces, and later recycled server eids.
    for (let i = 0; i < 7; i++) client.create();
    let body = server.create();
    const owner = server.create();
    server.add(body, Position, { value: [1, 2, 3], count: 0 });
    server.add(body, Replicate);
    server.add(owner, Replicate);
    server.add(owner, Target, { target: body });
    server.add(owner, Secret, { value: 123 });
    const link = seededLink(0x51a1107, { minDelay: 1, maxDelay: 8, loss: 0.18, duplication: 0.35 });
    const rules = [
        new Map([
            [1, [Position, Target]],
            [2, [Position]],
        ]),
    ];
    const sender = new ReplicationSender(server, schema, rules, 1, link.a);
    const expected = new Map<number, Map<number, number[]>>();
    const references = new Map<number, number[] | null | undefined>();
    const confirmed: number[] = [];
    const receiver = new ReplicationReceiver(client, schema, link.b, (tick, rows) => {
        confirmed.push(tick);
        check(server, client, receiver, rows, expected.get(tick)!, tick, references.get(tick));
    });
    try {
        for (let tick = 0; tick < 80; tick++) {
            link.advance(tick);
            if (tick === 24) server.destroy(body);
            if (tick === 25) {
                body = server.create();
                server.add(body, Position);
                server.add(body, Replicate);
                server.storage(Target).target.set(owner, body);
            }
            if (server.has(body, Position)) {
                server.storage(Position).value.set(body, tick / 4, tick, -tick);
                server.storage(Position).count.set(body, tick);
            }
            if (tick === 40) server.remove(owner, Target);
            if (tick === 41) server.add(owner, Target, { target: body });
            server.tick();
            if (tick >= join) {
                const snapshot = new Map<number, number[]>();
                for (const eid of server.query([Position])) {
                    const p = server.get(eid, Position)!;
                    snapshot.set(eid, [...p.value, p.count]);
                }
                const target = server.get(owner, Target)?.target;
                references.set(
                    tick,
                    target === undefined ? undefined : target === 0 ? null : snapshot.get(target)!,
                );
                expected.set(tick, snapshot);
                sender.record(tick);
                sender.flush();
                receiver.flush();
            }
        }
        for (let time = 80; time < 350; time++) {
            link.advance(time);
            sender.flush();
            receiver.flush();
        }
        expect(confirmed).toEqual(Array.from({ length: 80 - join }, (_, i) => i + join));
        expect(link.counts.dropped).toBeGreaterThan(0);
        expect(link.counts.duplicated).toBeGreaterThan(0);
    } finally {
        server.dispose();
        client.dispose();
    }
}

function compare(
    client: World,
    receiver: ReplicationReceiver,
    rows: ReplicatedRows,
    expected: Map<number, number[]>,
) {
    const actual: number[][] = [];
    expect(receiver.entities.size).toBe(rows.size);
    expect([...client.query([Position])].length).toBe(expected.size);
    for (const [id, row] of rows) {
        const eid = receiver.entities.get(id)!;
        const p = client.get(eid, Position);
        expect(client.has(eid, Position)).toBe(row.has(0));
        expect(client.has(eid, Target)).toBe(row.has(1));
        if (p) actual.push([...p.value, p.count]);
    }
    expect(actual).toEqual([...expected.values()]);
}

test("every confirmed tick equals authoritative rows through loss, duplication and reordering", () => {
    exercise(0, (_s, c, r, rows, expected) => compare(c, r, rows, expected));
});

test("entity references resolve in the receiver's eid space across despawn and eid reuse", () => {
    exercise(0, (_s, c, _r, _rows, _expected, _tick, authoritative) => {
        const owners = [...c.query([Target])];
        expect(owners.length).toBe(authoritative === undefined ? 0 : 1);
        if (authoritative === undefined) return;
        const target = c.get(owners[0]!, Target)!.target;
        if (authoritative === null) {
            expect(target).toBe(0);
            return;
        }
        expect(target).not.toBe(0);
        const position = c.get(target, Position)!;
        expect([...position.value, position.count]).toEqual(authoritative);
    });
});

test("an undisclosed component never arrives", () => {
    exercise(0, (_s, c, _r, rows) => {
        expect([...c.query([Secret])]).toEqual([]);
        for (const row of rows.values()) expect(row.has(2)).toBe(false);
    });
});

test("a late join confirms its image cut-over and every subsequent tick exactly once", () => {
    exercise(17, (_s, c, r, rows, expected) => compare(c, r, rows, expected));
});

test("values are resent against acknowledged fields and each recipient has its own disclosure", () => {
    const server = new World();
    const clients = [new World(), new World()];
    try {
        const eid = server.create();
        server.add(eid, Replicate);
        server.add(eid, Position, { value: [1, 2, 3] });
        server.add(eid, Secret, { value: 17 });
        const rules = [
            new Map([
                [1, [Position, Secret]],
                [2, [Position]],
            ]),
        ];
        for (let i = 0; i < 2; i++) {
            const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
            const sender = new ReplicationSender(server, schema, rules, i + 1, link.a);
            const receiver = new ReplicationReceiver(clients[i]!, schema, link.b);
            sender.record(0);
            sender.flush();
            receiver.flush();
            sender.flush();
            server.storage(Position).count.set(eid, 5);
            sender.record(1);
            sender.flush();
            const changed = link.b.receive().at(-1)!;
            sender.record(2);
            sender.flush();
            const resent = link.b
                .receive()
                .find((bytes) => new DataView(bytes.buffer).getUint32(8, true) === 2)!;
            expect(resent.length).toBe(changed.length);
            // Acknowledge the actual delivered tick before recording an unchanged value.
            link.a.send(changed);
            link.a.send(resent);
            receiver.flush();
            sender.flush();
            sender.record(3);
            sender.flush();
            const unchanged = link.b
                .receive()
                .find((bytes) => new DataView(bytes.buffer).getUint32(8, true) === 3)!;
            expect(unchanged.length).toBeLessThan(resent.length);
            expect([...clients[i]!.query([Secret])].length).toBe(i === 0 ? 1 : 0);
            server.storage(Position).count.set(eid, 0);
        }
    } finally {
        server.dispose();
        for (const client of clients) client.dispose();
    }
});

test("input before the image cannot acknowledge an unconfirmed tick", () => {
    const server = new World(),
        client = new World();
    try {
        const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
        const sender = new ReplicationSender(server, [], [], 1, link.a);
        const receiver = new ReplicationReceiver(client, [], link.b);
        sender.record(0);
        const legacyInput = new Uint8Array(4);
        const input = new Uint8Array([4, 0, 0, 0, 17]);
        link.b.send(legacyInput);
        link.b.send(input);
        const unhandled = sender.flush();
        expect(unhandled).toContainEqual(legacyInput);
        expect(unhandled).toContainEqual(input);
        const operation = new Uint8Array([5, 0, 0, 0, 42]);
        link.a.send(operation);
        expect(receiver.flush()).toEqual([operation]);
        expect(receiver.confirmed).toBe(0);
    } finally {
        server.dispose();
        client.dispose();
    }
});

test("declared widths round-trip signed integers, half floats, vectors and wire references", () => {
    const Values = component("NetworkTestWidths", {
        float: f32,
        half: f16,
        signed: i32,
        unsigned: u32,
        short: u16,
        byte: u8,
        vector: vec3,
        target: entity,
    });
    const server = new World(),
        client = new World();
    try {
        client.create();
        const eid = server.create();
        server.add(eid, Replicate);
        server.add(eid, Values, {
            float: -1.25,
            half: 0.333,
            signed: -2147483648,
            unsigned: 4294967295,
            short: 65535,
            byte: 255,
            vector: [1.5, -2.5, 3.5],
            target: eid,
        });
        const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
        const sender = new ReplicationSender(
            server,
            [Values],
            [new Map([[1, [Values]]])],
            1,
            link.a,
        );
        const receiver = new ReplicationReceiver(client, [Values], link.b);
        sender.record(5);
        sender.flush();
        const image = link.b.receive()[0]!;
        expect(image.length).toBe(109);
        link.a.send(image);
        receiver.flush();
        sender.flush();
        const local = receiver.entities.values().next().value!;
        expect(client.get(local, Values)).toEqual({ ...server.get(eid, Values)!, target: local });
        expect(() => sender.record(7)).toThrow("consecutive");
        sender.record(6);
        sender.flush();
        const frame = link.b.receive()[0]!;
        expect(frame.length).toBe(20);
        link.a.send(frame);
        receiver.flush();
        expect(receiver.confirmed).toBe(6);
    } finally {
        server.dispose();
        client.dispose();
    }
});

function frameSizes(entityCount: number) {
    const server = new World(),
        client = new World();
    try {
        let first = 0;
        for (let i = 0; i < entityCount; i++) {
            const eid = server.create();
            if (i === 0) first = eid;
            server.add(eid, Replicate);
            server.add(eid, Position);
            server.add(eid, Secret);
        }
        const schema = [Position, Secret];
        const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
        const sender = new ReplicationSender(server, schema, [new Map([[1, schema]])], 1, link.a);
        const receiver = new ReplicationReceiver(client, schema, link.b);
        function deliver(tick: number) {
            sender.record(tick);
            sender.flush();
            const bytes = link.b.receive()[0]!;
            link.a.send(bytes);
            receiver.flush();
            sender.flush();
            expect(receiver.confirmed).toBe(tick);
            return bytes.length;
        }
        deliver(0);
        const unchanged = deliver(1);
        server.storage(Secret).value.set(first, 1.5);
        const scalar = deliver(2);
        const local = receiver.entities.values().next().value!;
        expect(client.get(local, Secret)!.value).toBe(1.5);
        return { unchanged, scalar };
    } finally {
        server.dispose();
        client.dispose();
    }
}

test("unchanged frames have constant size at 1, 10 and 100 entities", () => {
    expect([1, 10, 100].map((count) => frameSizes(count).unchanged)).toEqual([20, 20, 20]);
});

test("one changed scalar adds only its entity, component, field index and value", () => {
    const sizes = [1, 10, 100].map(frameSizes);
    expect(sizes.map((size) => size.scalar)).toEqual([44, 44, 44]);
    expect(sizes.map((size) => size.scalar - size.unchanged)).toEqual([24, 24, 24]);
});
