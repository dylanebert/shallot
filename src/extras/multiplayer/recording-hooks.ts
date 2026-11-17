import type { ReplicationReceiver } from ".";
import type { InputRecord } from "./intent";
import type { ClientPrediction } from "./prediction";

/** @internal Per-World recording hooks, absent unless recording or replay was requested. */
export const PredictionRecording = {
    create: (): {
        receiver?: (receiver: ReplicationReceiver) => void;
        attach?: (prediction: ClientPrediction) => void;
        sample?: (prediction: ClientPrediction, tick: number) => InputRecord | undefined;
    } => ({}),
};
