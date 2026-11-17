import type { InputRecord } from "./intent";
import type { ClientPrediction } from "./prediction";

/** @internal Per-World recording hooks, absent unless recording or replay was requested. */
export const PredictionRecording = {
    create: (): {
        attach?: (prediction: ClientPrediction) => void;
        sample?: (prediction: ClientPrediction, tick: number) => InputRecord | undefined;
    } => ({}),
};
