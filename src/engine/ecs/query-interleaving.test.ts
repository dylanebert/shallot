import { expect, test } from "bun:test";
import { RegisteredQuery } from "./query";

test("removal after a nested iteration never skips an untouched outer member", () => {
    const query = new RegisteredQuery([]);
    for (const eid of [1, 2, 3, 4]) query.add(eid);
    const outer = query[Symbol.iterator]();
    expect(outer.next().value).toBe(1);
    query.add(5);
    expect([...query]).toEqual([1, 2, 3, 4, 5]);
    query.remove(3);
    const rest: number[] = [];
    for (let next = outer.next(); !next.done; next = outer.next()) rest.push(next.value);
    expect(rest).toEqual([2, 4]);
    expect([...query]).toEqual([1, 2, 4, 5]);
});

test("fixed-seed interleaved mutation and nested iterators agree with a membership-entry reference model", () => {
    let seed = 0x6d2b79f5;
    function random(n: number) {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        return (seed >>> 0) % n;
    }
    for (let round = 0; round < 64; round++) {
        const query = new RegisteredQuery([]);
        type Entry = { eid: number; alive: boolean };
        const members = new Map<number, Entry>();
        let order: Entry[] = [];
        let active = 0;
        function add(eid: number) {
            query.add(eid);
            if (members.has(eid)) return;
            const entry = { eid, alive: true };
            members.set(eid, entry);
            order.push(entry);
        }
        function remove(eid: number) {
            query.remove(eid);
            const entry = members.get(eid);
            if (entry) entry.alive = false;
            members.delete(eid);
        }
        function start() {
            if (!active) order = order.filter((entry) => entry.alive).sort((a, b) => a.eid - b.eid);
            const entries = order.slice();
            active++;
            let index = 0;
            let running = true;
            const actual = query[Symbol.iterator]();
            const finish = () => {
                if (running) active--;
                running = false;
            };
            return {
                next() {
                    while (running && index < entries.length) {
                        const entry = entries[index++];
                        if (!entry.alive) continue;
                        expect(actual.next()).toEqual({ value: entry.eid, done: false });
                        return false;
                    }
                    finish();
                    expect(actual.next().done).toBe(true);
                    return true;
                },
                return() {
                    finish();
                    actual.return!();
                },
            };
        }
        for (const eid of [4, 1, 3, 2]) add(eid);
        const iterators: ReturnType<typeof start>[] = [];
        for (let action = 0; action < 128; action++) {
            const op = random(5);
            if (op === 0) add(random(12) + 1);
            else if (op === 1) remove(random(12) + 1);
            else if (op === 2 && iterators.length < 4) iterators.push(start());
            else if (iterators.length) {
                const index = random(iterators.length);
                const iterator = iterators[index];
                if (op === 4) {
                    iterator.return();
                    iterators.splice(index, 1);
                } else if (iterator.next()) iterators.splice(index, 1);
            }
        }
        for (const iterator of iterators) while (!iterator.next()) {}
        expect([...query]).toEqual([...members.keys()].sort((a, b) => a - b));
    }
});
