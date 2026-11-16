import { createConnection, createServer, type Socket } from "node:net";

/** TCP stream controls, following Toxiproxy's directional latency and bandwidth toxics.
 * Times are milliseconds; bandwidth is bytes/second, with zero disabling the limit.
 * A stall retains bytes until released, unlike Toxiproxy's byte-discarding timeout toxic.
 * Changes affect queued traffic except latency/jitter, sampled when each chunk arrives.
 * Jitter is uniform ±jitter, clamped at zero delay. Stream order is preserved. */
export interface StreamFaults {
    latency: number;
    jitter: number;
    bandwidth: number;
    stalled: boolean;
}

/** Per-peer TCP impairment. Upstream is client → server; downstream is server → client.
 * Controls start disabled, after the WebSocket handshake. No bytes are dropped to model loss:
 * packet loss beneath TCP requires a packet-level tool, not a stream proxy.
 * Seeded jitter repeats draws, not host chunk boundaries or IO timing. */
export interface PeerProxy {
    upstream: StreamFaults;
    downstream: StreamFaults;
    /** Abruptly resets both TCP legs, closing this peer rather than clearing its controls. */
    resetPeer(): void;
}

// Count WebSocket payloads at the independent TCP boundary, excluding HTTP and frame headers.
function payloadCounter() {
    const counts = { bytes: 0, messages: 0 };
    let pending = Buffer.alloc(0);
    let upgraded = false;
    return {
        counts,
        push(bytes: Buffer) {
            pending = Buffer.concat([pending, bytes]);
            if (!upgraded) {
                const end = pending.indexOf("\r\n\r\n");
                if (end < 0) return;
                pending = pending.subarray(end + 4);
                upgraded = true;
            }
            while (pending.length >= 2) {
                let length = pending[1]! & 127;
                let header = 2;
                if (length === 126) {
                    if (pending.length < 4) return;
                    length = pending.readUInt16BE(2);
                    header = 4;
                } else if (length === 127) {
                    if (pending.length < 10) return;
                    length = Number(pending.readBigUInt64BE(2));
                    header = 10;
                }
                if (pending[1]! & 128) header += 4;
                if (pending.length < header + length) return;
                const opcode = pending[0]! & 15;
                if (opcode < 8) {
                    counts.bytes += length;
                    if (opcode !== 0) counts.messages++;
                }
                pending = pending.subarray(header + length);
            }
        },
    };
}

/** @internal Loopback-only TCP proxy owned by a test session. */
export async function createProxy(port: number, seed: number) {
    const faults = (): StreamFaults => ({ latency: 0, jitter: 0, bandwidth: 0, stalled: false });
    const sockets = new Set<Socket>();
    const counters = { upstream: payloadCounter(), downstream: payloadCounter() };
    const pumps = new Set<() => void>();
    let error: unknown;
    const resetPeer = () => {
        for (const socket of sockets) {
            if (socket.connecting) socket.destroy();
            else if (!socket.destroyed) socket.resetAndDestroy();
        }
        for (const stop of pumps) stop();
    };
    const controls: PeerProxy = {
        upstream: faults(),
        downstream: faults(),
        resetPeer,
    };
    const check = () => {
        if (error) throw error;
        for (const stream of [controls.upstream, controls.downstream]) {
            for (const value of [stream.latency, stream.jitter, stream.bandwidth])
                if (!Number.isFinite(value) || value < 0)
                    throw new Error(
                        "Proxy latency, jitter and bandwidth must be finite and nonnegative",
                    );
            if (typeof stream.stalled !== "boolean")
                throw new Error("Proxy stalled must be boolean");
        }
    };
    const track = (socket: Socket) => {
        sockets.add(socket);
        socket.on("error", resetPeer);
        socket.once("close", () => sockets.delete(socket));
        socket.setNoDelay(true);
    };
    const pipe = (
        source: Socket,
        destination: Socket,
        direction: "upstream" | "downstream",
        state: number,
    ) => {
        const queue: { bytes: Buffer; due: number }[] = [];
        let timer: ReturnType<typeof setTimeout> | undefined;
        let queued = 0,
            previous = performance.now(),
            credit = 0,
            blocked = false,
            ended = false,
            stopped = false;
        const stop = () => {
            stopped = true;
            clearTimeout(timer);
            queue.length = 0;
            pumps.delete(stop);
        };
        pumps.add(stop);
        const pump = () => {
            if (stopped) return;
            timer = undefined;
            try {
                check();
                const now = performance.now();
                const stream = controls[direction];
                if (stream.stalled || !queue.length) credit = 0;
                else credit += ((now - previous) * stream.bandwidth) / 1000;
                previous = now;
                while (!stream.stalled && !blocked && queue.length && queue[0]!.due <= now) {
                    const chunk = queue[0]!;
                    const length = stream.bandwidth
                        ? Math.min(chunk.bytes.length, Math.floor(credit))
                        : chunk.bytes.length;
                    if (!length) break;
                    const bytes = chunk.bytes.subarray(0, length);
                    chunk.bytes = chunk.bytes.subarray(length);
                    queued -= length;
                    if (stream.bandwidth) credit -= length;
                    if (!chunk.bytes.length) queue.shift();
                    blocked = !destination.write(bytes);
                }
                if (queued < 65536 && !source.destroyed) source.resume();
                if (!queue.length && ended) {
                    destination.end();
                    stop();
                } else if (queue.length) timer = setTimeout(pump, 1);
            } catch (cause) {
                error = cause;
                resetPeer();
            }
        };
        source.on("data", (bytes: Buffer) => {
            if (stopped) return;
            counters[direction].push(bytes);
            const stream = controls[direction];
            if (stream.jitter) state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            const delay = Math.max(
                0,
                stream.latency + ((state / 0x100000000) * 2 - 1) * stream.jitter,
            );
            if (!queue.length) {
                previous = performance.now();
                credit = 0;
            }
            queue.push({ bytes, due: performance.now() + delay });
            queued += bytes.length;
            if (queued >= 65536) source.pause();
            if (!timer) pump();
        });
        destination.on("drain", () => {
            blocked = false;
            if (!timer) pump();
        });
        source.on("end", () => {
            ended = true;
            if (!timer) pump();
        });
        // A source can finish both TCP halves while its read bytes still await delivery here.
        // Only this direction's drained queue forwards FIN; a clean close must not stop its pump.
        source.once("close", () => {
            if (!ended) resetPeer();
        });
        destination.once("close", () => {
            if (!destination.writableFinished) resetPeer();
        });
    };
    const listener = createServer({ allowHalfOpen: true }, (client) => {
        track(client);
        const upstream = createConnection({ host: "127.0.0.1", port, allowHalfOpen: true });
        track(upstream);
        pipe(client, upstream, "upstream", seed);
        pipe(upstream, client, "downstream", seed ^ 0x9e3779b9);
    });
    await new Promise<void>((resolve, reject) => {
        listener.once("error", reject);
        listener.listen(0, "127.0.0.1", resolve);
    });
    const close = listener.close.bind(listener);
    return {
        controls,
        counts: { upstream: counters.upstream.counts, downstream: counters.downstream.counts },
        port: (listener.address() as { port: number }).port,
        check,
        stop() {
            resetPeer();
            listener.close();
        },
        forceClose() {
            resetPeer();
            if (listener.listening) close();
        },
        get pendingSockets() {
            return sockets.size;
        },
    };
}

/** @internal Only connection refusal establishes listener release. */
export async function listenerReleased(port: number): Promise<boolean> {
    return new Promise((resolve, reject) => {
        const probe = createConnection({ host: "127.0.0.1", port });
        probe.once("connect", () => {
            probe.destroy();
            resolve(false);
        });
        probe.once("error", (error: NodeJS.ErrnoException) => {
            if (error.code === "ECONNREFUSED") resolve(true);
            else reject(error);
        });
    });
}
