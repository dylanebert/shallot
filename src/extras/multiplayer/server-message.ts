import { encodeText, readText, writeText } from "../../engine/ecs";
import { requireCanonical, writeFloat64 } from "./canonical";
import { MAX_MESSAGE_BYTES } from "./link";

export interface ServerMessage {
    /** Game-defined unsigned 32-bit topic. */
    topic: number;
    /** Numeric payload. Together with the topic, header and text it must fit 4096 bytes. */
    values: readonly number[];
    /** Optional UTF-8 payload, at most 1024 bytes. */
    text?: string;
}

export const ServerMessageKind = 8;
const TextCapacity = 1024;
const HeaderBytes = 10;

/** @internal LE u32 kind and topic, u16 lane count, float64 lanes, optional u16 size and UTF-8. */
export function encodeServerMessage({ topic, values, text }: ServerMessage): Uint8Array {
    if (!Number.isInteger(topic) || topic < 0 || topic > 0xffffffff)
        throw new Error("Server message topic must be a u32");
    const payload = text === undefined ? undefined : encodeText(text, TextCapacity);
    const end = HeaderBytes + 8 * values.length;
    const size = end + (payload ? 2 + payload.length : 0);
    if (size > MAX_MESSAGE_BYTES) throw new Error("Server message exceeds frame cap");
    const bytes = new Uint8Array(size);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, ServerMessageKind, true);
    view.setUint32(4, topic, true);
    view.setUint16(8, values.length, true);
    for (let i = 0; i < values.length; i++) writeFloat64(view, HeaderBytes + 8 * i, values[i]!);
    if (payload) writeText(view, end, payload);
    return bytes;
}

/** @internal Refuse any bytes the encoder would not produce. */
export function decodeOneOffMessage(bytes: Uint8Array, view: DataView): ServerMessage {
    if (bytes.length < HeaderBytes || bytes.length > MAX_MESSAGE_BYTES)
        throw new Error("Invalid server message size");
    const topic = view.getUint32(4, true);
    const count = view.getUint16(8, true);
    const end = HeaderBytes + 8 * count;
    if (end > bytes.length) throw new Error("Truncated server message values");
    const values = Array.from({ length: count }, (_, i) =>
        view.getFloat64(HeaderBytes + 8 * i, true),
    );
    const message =
        bytes.length === end
            ? { topic, values }
            : { topic, values, text: readText(view, end, TextCapacity) };
    requireCanonical(bytes, encodeServerMessage(message));
    return message;
}
