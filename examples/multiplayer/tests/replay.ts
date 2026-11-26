import {
    type RecordingCall,
    replaySession,
    type SessionRecording,
} from "@dylanebert/shallot/multiplayer";
import { GAME } from "../src/network";

const recordings: SessionRecording[] = await Bun.file(process.argv[2]!).json();
for (const recording of recordings) await replaySession(GAME, "multiplayer-1", recording);
console.log(`${recordings.length} recording(s) replayed through every hash`);

// A replay that passes without its received state would be vacuous: dropping the client's last
// incoming state must fail at that tick or the next fixed boundary.
const recording = recordings.find((recording) => recording.role === "client")!;
const flatten = (calls: RecordingCall[]): RecordingCall[] =>
    calls.flatMap((call) => [call, ...flatten(call.calls)]);
const received = flatten(recording.calls)
    .reverse()
    .find(
        (call) =>
            call.kind === "receive" &&
            (call.result as number[][]).some((bytes) => bytes.length > 16),
    );
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
console.log(`dropping the client's receive at tick ${received.tick} fails there`);
