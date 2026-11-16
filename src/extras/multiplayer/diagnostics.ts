import { Diagnostic, DiagnosticsStore, type World } from "../../engine";
import type { Connection } from "./link";

// Transport receipt precedes the World's next drain. Keeping it on the owned message avoids
// confusing the answering end's scheduling delay with network round trip.
const receipt = Symbol("network-receipt");
export const transportClock = Symbol("network-clock");
type TimedBytes = Uint8Array & { [receipt]?: number };
type TimedConnection = Connection & { [transportClock]?: () => number };
export function stampReceipt(bytes: Uint8Array, milliseconds: number): Uint8Array {
    (bytes as TimedBytes)[receipt] = milliseconds;
    return bytes;
}

function word(bytes: Uint8Array, offset: number): number {
    return (
        (bytes[offset]! |
            (bytes[offset + 1]! << 8) |
            (bytes[offset + 2]! << 16) |
            (bytes[offset + 3]! << 24)) >>>
        0
    );
}

// Little-endian ping: u32 kind (6), u32 ID. Pong: u32 kind (7), u32 ID,
// f64 remote receipt-to-reply milliseconds. Neither message carries a remote clock epoch.
const Ping = 6;
const Pong = 7;
const Interval = 100;
const Capacity = 1024;
const Observers = {
    create: (): {
        connections: WeakMap<Connection, NetworkDiagnostics>;
        client?: NetworkDiagnostics;
    } => ({ connections: new WeakMap() }),
};

/** @internal Resolve a session recorder through game-owned client routing wrappers. */
export function networkDiagnostics(
    world: World,
    connection: Connection,
): NetworkDiagnostics | undefined {
    const observers = world.resource(Observers);
    return observers.connections.get(connection) ?? observers.client;
}

/** @internal One recorder per connection lifetime. Does not own the transport. */
export function observeNetwork(
    world: World,
    connection: Connection,
    prefix = "network/",
): NetworkDiagnostics {
    const observers = world.resource(Observers);
    let observer = observers.connections.get(connection);
    // A client World has one upstream; game routing wrappers share its transport recorder.
    if (!observer && prefix === "network/") observer = observers.client;
    if (!observer) {
        observer = new NetworkDiagnostics(world, connection, prefix);
        observers.connections.set(connection, observer);
        if (prefix === "network/") observers.client = observer;
    }
    return observer;
}

/** Local-end counters include protocol traffic and retries, but not transport framing.
 * Ping/pong follows lightyear's PingManager: elapsed local time minus remote receipt-to-reply
 * time. The local receipt timestamp excludes local drain scheduling too. Samples are matched
 * by ID, including out-of-order pongs; the bounded outstanding ring expires after 1024 pings.
 */
class NetworkDiagnostics {
    private readonly _store: DiagnosticsStore;
    private readonly _rows = new Map<string, Diagnostic>();
    private readonly _clock: () => number;
    private readonly _send: Connection["send"];
    private readonly _receive: Connection["receive"];
    private readonly _ids = new Float64Array(Capacity).fill(-1);
    private readonly _sent = new Float64Array(Capacity);
    private readonly _ping = new Uint8Array(8);
    private readonly _pong = new Uint8Array(16);
    private readonly _pingView = new DataView(this._ping.buffer);
    private readonly _pongView = new DataView(this._pong.buffer);
    private readonly _incoming = new Uint8Array(16);
    private readonly _incomingView = new DataView(this._incoming.buffer);
    private _next = 0;
    private _lastPing = -Infinity;
    private _lastReceive: number;
    private _received = false;
    private _bytesSent = 0;
    private _bytesReceived = 0;
    private _messagesSent = 0;
    private _messagesReceived = 0;
    private _pending: Uint8Array[] | undefined;
    private readonly _empty: Uint8Array[] = [];
    readonly connection: Connection;
    constructor(world: World, connection: Connection, prefix: string) {
        this.connection = connection;
        this._store = world.resource(DiagnosticsStore);
        this._clock = (connection as TimedConnection)[transportClock] ?? (() => performance.now());
        this._lastReceive = this._clock();
        this._send = connection.send.bind(connection);
        this._receive = connection.receive.bind(connection);
        this._pingView.setUint32(0, Ping, true);
        this._pongView.setUint32(0, Pong, true);
        for (const [name, suffix] of [
            ["round_trip", "ms"],
            ["bytes_sent", "B"],
            ["bytes_received", "B"],
            ["messages_sent", ""],
            ["messages_received", ""],
            ["confirmed_tick_lag", "ticks"],
            ["time_since_receive", "ms"],
        ] as const) {
            const row = new Diagnostic(prefix + name, { suffix });
            this._rows.set(name, row);
            this._store.add(row);
        }
        connection.send = (bytes) => {
            if (connection.closed) return this._send(bytes);
            this._send(bytes);
            this._bytesSent += bytes.byteLength;
            this._messagesSent++;
            this.record("bytes_sent", this._bytesSent);
            this.record("messages_sent", this._messagesSent);
        };
        connection.receive = () => {
            const empty = this.poll();
            const messages = this._pending ?? empty;
            this._pending = undefined;
            return messages;
        };
        for (const name of ["bytes_sent", "bytes_received", "messages_sent", "messages_received"])
            this.record(name, 0);
    }
    record(name: string, value: number): void {
        this._rows.get(name)!.addMeasurement(this._clock() / 1000, value);
    }
    poll(): Uint8Array[] {
        const messages = this._receive();
        const now = this._clock();
        let count = 0;
        for (let i = 0; i < messages.length; i++) {
            const bytes = messages[i]!;
            const arrived = (bytes as TimedBytes)[receipt] ?? now;
            this._lastReceive = this._received ? Math.max(this._lastReceive, arrived) : arrived;
            this._received = true;
            this._bytesReceived += bytes.byteLength;
            this._messagesReceived++;
            const kind = bytes.length >= 4 ? word(bytes, 0) : 0;
            if (kind === Ping && bytes.length === 8) {
                this._pongView.setUint32(4, word(bytes, 4), true);
                this._pongView.setFloat64(8, Math.max(0, this._clock() - arrived), true);
                this.connection.send(this._pong);
            } else if (kind === Pong && bytes.length === 16) {
                const id = word(bytes, 4);
                const slot = id % Capacity;
                this._incoming.set(bytes);
                const held = this._incomingView.getFloat64(8, true);
                if (this._ids[slot] === id && Number.isFinite(held) && held >= 0) {
                    this.record("round_trip", Math.max(0, arrived - this._sent[slot]! - held));
                    this._ids[slot] = -1;
                }
            } else messages[count++] = bytes;
        }
        if (messages.length) {
            this.record("bytes_received", this._bytesReceived);
            this.record("messages_received", this._messagesReceived);
        }
        if (count) {
            messages.length = count;
            if (this._pending) for (let i = 0; i < count; i++) this._pending.push(messages[i]!);
            else this._pending = messages;
        }
        this.record("time_since_receive", Math.max(0, now - this._lastReceive));
        if (!this.connection.closed && now - this._lastPing >= Interval) {
            const id = this._next;
            this._next = (id + 1) >>> 0;
            const slot = id % Capacity;
            this._ids[slot] = id;
            this._sent[slot] = this._lastPing = now;
            this._pingView.setUint32(4, id, true);
            this.connection.send(this._ping);
        }
        return count ? messages : this._empty;
    }
    remove(): void {
        for (const row of this._rows.values()) this._store.remove(row.path);
        this._pending = undefined;
    }
}
