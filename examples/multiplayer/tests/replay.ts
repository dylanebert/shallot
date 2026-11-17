import {
    type RecordingCall,
    replaySession,
    type SessionRecording,
} from "@dylanebert/shallot/multiplayer";
import { GAME } from "../src/network";

const recordings: SessionRecording[] = await Bun.file(process.argv[2]!).json();
if (process.argv.includes("--drop-receive")) {
    const recording = recordings.find((recording) => recording.role === "client")!;
    const flatten = (calls: RecordingCall[]): RecordingCall[] =>
        calls.flatMap((call) => [call, ...flatten(call.calls)]);
    const received = flatten(recording.calls)
        .reverse()
        .find(
            (call) =>
                call.kind === "receive" &&
                (call.result as number[][]).some((bytes) => bytes.length > 16),
        )!;
    if (!received) throw new Error("No incoming state recorded");
    received.result = [];
    let failure = "";
    try {
        await replaySession(GAME, "multiplayer-1", recording);
    } catch (error) {
        failure = String(error);
    }
    if (
        !failure.includes(`fixed tick ${received.tick}:`) &&
        !failure.includes(`fixed tick ${received.tick + 1}:`)
    )
        throw new Error(`Missing receive at ${received.tick} did not fail there: ${failure}`);
} else {
    for (const recording of recordings) await replaySession(GAME, "multiplayer-1", recording);
}
