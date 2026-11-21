import { stampReceipt, transportClock } from "./diagnostics";

/** Largest message a client sends, in bytes: Colyseus's Bun transport default, which
 * serveGame and serveRooms refuse beyond. */
export const MAX_MESSAGE_BYTES = 4 * 1024;

/** A byte transport. Admission transfers its lifetime to the server connection entity. */
export interface Connection {
    /** URL supplying received bytes; wrappers preserve it, in-process and replayed links omit it. */
    readonly endpoint?: string;
    /** True once this transport cannot deliver again. Omitted by transports without status polling. */
    readonly closed?: boolean;
    /** Changes when a replacement stream starts, before any of its bytes are delivered. */
    readonly generation?: number;
    /** Idempotent, non-throwing release: close the transport, discard queued bytes and refuse
     * future delivery. Closing either end of a seeded link closes both ends. */
    close(): void;
    /** Reports a terminal protocol error after the receiver closes this connection. */
    onError?(error: Error): void;
    send(bytes: Uint8Array): void;
    receive(): Uint8Array[];
}

const failedServerConnections = new WeakSet<Connection>();

/** @internal A protocol failure is terminal, including on reconnectable transports. */
export function serverConnectionFailed(connection: Connection): boolean {
    return failedServerConnections.has(connection);
}

/** @internal Close and report a server protocol failure at most once. */
export function failServerConnection(connection: Connection, error: Error): void {
    if (serverConnectionFailed(connection)) return;
    failedServerConnections.add(connection);
    connection.close();
    connection.onError?.(error);
}

export interface LinkFaults {
    /** Delay in caller-supplied clock units. */
    minDelay: number;
    maxDelay: number;
    loss: number;
    duplication: number;
}

/** Seeded byte transport. advance sets a transport clock, never ticks a World. */
export function seededLink(seed: number, faults: LinkFaults) {
    if (
        faults.minDelay < 0 ||
        faults.maxDelay < faults.minDelay ||
        faults.loss < 0 ||
        faults.loss > 1 ||
        faults.duplication < 0 ||
        faults.duplication > 1
    )
        throw new Error("Invalid link faults");
    let now = 0;
    let closed = false;
    const queues: { due: number; order: number; bytes: Uint8Array }[][] = [[], []];
    const counts = { sent: 0, dropped: 0, duplicated: 0 };
    const random = () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 0x100000000;
    };
    const ends: Connection[] = [0, 1].map((side) => ({
        [transportClock]: () => now,
        get closed() {
            return closed;
        },
        close() {
            closed = true;
            queues[0]!.length = 0;
            queues[1]!.length = 0;
        },
        send(bytes: Uint8Array) {
            if (closed) return;
            counts.sent++;
            if (random() < faults.loss) {
                counts.dropped++;
                return;
            }
            const copies = random() < faults.duplication ? 2 : 1;
            counts.duplicated += copies - 1;
            for (let i = 0; i < copies; i++)
                queues[1 - side]!.push({
                    due: now + faults.minDelay + random() * (faults.maxDelay - faults.minDelay),
                    order: random(),
                    bytes: bytes.slice(),
                });
        },
        receive() {
            const queue = queues[side]!;
            const ready = queue
                .filter((p) => p.due <= now)
                .sort((a, b) => a.due - b.due || a.order - b.order);
            queues[side] = queue.filter((p) => p.due > now);
            return ready.map((p) => stampReceipt(p.bytes, p.due));
        },
    }));
    return {
        a: ends[0]!,
        b: ends[1]!,
        counts,
        advance(time: number) {
            if (time < now) throw new Error("Link clock cannot go backwards");
            now = time;
        },
    };
}
