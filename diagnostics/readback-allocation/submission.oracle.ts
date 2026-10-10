import { setupGlobals } from "@dylanebert/shallot/webgpu";
import createSubject from "./render.entry";

await setupGlobals();

const TRIALS = 5;
const BATCH_FRAMES = 60;

type SubmissionSample = {
    adapter: string;
    adapterClass: string;
    firstDrainMilliseconds: number;
    nextDrainMilliseconds: number;
};

async function measure(): Promise<SubmissionSample> {
    const subject = await createSubject();
    const device = subject.world.gpu.device;
    const lost = device.lost.then((info) => {
        throw new Error(`render allocation device lost: ${info.message}`);
    });
    void lost.catch(() => {});
    let validation: GPUError | undefined;
    let failValidation!: (error: GPUError) => void;
    const failed = new Promise<never>((_, reject) => {
        failValidation = reject;
    });
    void failed.catch(() => {});
    const onError = (event: GPUUncapturedErrorEvent) => {
        validation ??= event.error;
        failValidation(validation);
    };
    device.addEventListener("uncapturederror", onError);

    const drain = async (): Promise<number> => {
        const start = performance.now();
        await Promise.race([device.queue.onSubmittedWorkDone(), failed, lost]);
        if (validation) throw validation;
        return performance.now() - start;
    };
    try {
        for (let frame = 0; frame < BATCH_FRAMES; frame++) subject.step();
        const firstDrainMilliseconds = await drain();
        for (let frame = 0; frame < BATCH_FRAMES; frame++) subject.step();
        const nextDrainMilliseconds = await drain();
        return {
            adapter: subject.world.gpu.adapter.identity,
            adapterClass: subject.world.gpu.adapter.class,
            firstDrainMilliseconds,
            nextDrainMilliseconds,
        };
    } finally {
        device.removeEventListener("uncapturederror", onError);
        subject.dispose();
    }
}

const samples: SubmissionSample[] = [];
for (let trial = 0; trial < TRIALS; trial++) samples.push(await measure());
const median = (values: number[]) =>
    values.sort((a, b) => a - b)[Math.floor(values.length / 2)]!;
console.info(
    JSON.stringify({
        oracle: "readback-allocation-frame-submissions",
        host: `${process.platform}/${process.arch}`,
        adapter: samples[0]!.adapter,
        adapterClass: samples[0]!.adapterClass,
        firstDrainFrames: BATCH_FRAMES + 1,
        nextDrainFrames: BATCH_FRAMES,
        firstDrainMilliseconds: samples.map(({ firstDrainMilliseconds }) => firstDrainMilliseconds),
        medianFirstDrainMilliseconds: median(
            samples.map(({ firstDrainMilliseconds }) => firstDrainMilliseconds),
        ),
        nextDrainMilliseconds: samples.map(({ nextDrainMilliseconds }) => nextDrainMilliseconds),
        medianNextDrainMilliseconds: median(
            samples.map(({ nextDrainMilliseconds }) => nextDrainMilliseconds),
        ),
    }),
);
