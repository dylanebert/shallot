import { createApp } from "../../engine";
import { observeNetwork, stampReceipt, transportClock } from "./diagnostics";

export let controlSink: object;
export function control() {
    controlSink = {};
}
export default async function create() {
    const app = await createApp({ defaults: false, plugins: [] });
    let now = 0;
    const ping = stampReceipt(new Uint8Array(8), 0);
    const pong = stampReceipt(new Uint8Array(16), 0);
    const pingView = new DataView(ping.buffer),
        pongView = new DataView(pong.buffer);
    pingView.setUint32(0, 6, true);
    pongView.setUint32(0, 7, true);
    const payload = new Uint8Array(24);
    const messages: Uint8Array[] = [];
    const empty: Uint8Array[] = [];
    const connection = {
        [transportClock]: () => now,
        close() {},
        send(bytes: Uint8Array) {
            if (bytes.length === 8) {
                pongView.setUint32(
                    4,
                    (bytes[4]! | (bytes[5]! << 8) | (bytes[6]! << 16) | (bytes[7]! << 24)) >>> 0,
                    true,
                );
            }
        },
        receive() {
            if (!messages.length) return empty;
            return messages;
        },
    };
    const diagnostics = observeNetwork(app.world, connection);
    return {
        step() {
            now += 100;
            stampReceipt(ping, now - 10);
            stampReceipt(pong, now);
            messages[0] = ping;
            messages[1] = pong;
            messages[2] = payload;
            const received = connection.receive();
            if (received.length !== 1 || received[0] !== payload) throw new Error("lost payload");
            diagnostics.record("confirmed_tick_lag", 2);
        },
        dispose: () => app.dispose(),
    };
}
