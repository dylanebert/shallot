import { openBasisImage } from "../../src/extras/gltf/basisu";
import { equal, modelFiles, SAMPLE_BATCHES, type Input, type Reference } from "./comparison";

interface Split { total: number; setup: number; teardown: number; copies: number; etc1: number; "etc2-rgba": number; astc: number }
const zero = (): Split => ({ total: 0, setup: 0, teardown: 0, copies: 0, etc1: 0, "etc2-rgba": 0, astc: 0 });
const minima = (samples: Split[]): Split => Object.fromEntries(Object.keys(zero()).map((key) => [key, Math.min(...samples.map((s) => s[key as keyof Split]))])) as unknown as Split;

export function attribution(reference: Reference, inputs: Input[]) {
    for (const model of ["StainedGlassLamp", "AnisotropyBarnLamp"]) {
        const files = modelFiles(reference, inputs, model);
        // Validate the no-supercompression control through both JavaScript entries.
        for (const f of files) {
            if (!f.plain) throw new Error(`${f.path}: missing plain control`);
            const original = openBasisImage(f.bytes);
            const plain = openBasisImage(f.plain);
            const ref = new reference.KTX2File(f.plain);
            if (!ref.startTranscoding()) throw new Error("reference start failed");
            try {
                for (let level = 0; level < f.levels; level++) {
                    const output = new Uint8Array(ref.getImageTranscodedSizeInBytes(level,0,0,f.format));
                    if (!ref.transcodeImage(output,level,0,0,f.format,0,-1,-1)) throw new Error("reference transcode failed");
                    if (!equal(original.transcode(level,f.target), output) || !equal(plain.transcode(level,f.target), output)) throw new Error("plain KTX2 control changed output");
                }
            } finally { original.dispose(); plain.dispose(); ref.close(); ref.delete(); }
        }
        let copyTime = 0;
        const set = Uint8Array.prototype.set;
        const slice = Uint8Array.prototype.slice;
        function run(own: boolean, plain: boolean): Split {
            const split = zero();
            copyTime = 0;
            Uint8Array.prototype.set = function (source, offset) {
                const start = performance.now();
                set.call(this,source,offset);
                copyTime += performance.now()-start;
            };
            Uint8Array.prototype.slice = function (start, end) {
                const before = performance.now();
                const value = slice.call(this,start,end);
                copyTime += performance.now()-before;
                return value;
            };
            const started = performance.now();
            try {
                for (const f of files) {
                    const bytes = plain ? f.plain! : f.bytes;
                    let start = performance.now();
                    let beforeCopies = copyTime;
                    if (own) {
                        const image = openBasisImage(bytes);
                        split.setup += performance.now()-start-(copyTime-beforeCopies);
                        try {
                            start = performance.now(); beforeCopies = copyTime;
                            for (let level = 0; level < f.levels; level++) image.transcode(level,f.target);
                            split[f.target] += performance.now()-start-(copyTime-beforeCopies);
                        } finally { const start=performance.now(); image.dispose(); split.teardown+=performance.now()-start; }
                    } else {
                        const image = new reference.KTX2File(bytes);
                        if (!image.startTranscoding()) throw new Error("reference start failed");
                        split.setup += performance.now()-start-(copyTime-beforeCopies);
                        try {
                            start = performance.now(); beforeCopies = copyTime;
                            for (let level = 0; level < f.levels; level++) {
                                const output = new Uint8Array(image.getImageTranscodedSizeInBytes(level,0,0,f.format));
                                if (!image.transcodeImage(output,level,0,0,f.format,0,-1,-1)) throw new Error("reference transcode failed");
                            }
                            split[f.target] += performance.now()-start-(copyTime-beforeCopies);
                        } finally { const start=performance.now(); image.close(); image.delete(); split.teardown+=performance.now()-start; }
                    }
                }
                split.copies = copyTime;
                split.total = performance.now() - started;
                return split;
            } finally { Uint8Array.prototype.set = set; Uint8Array.prototype.slice = slice; }
        }
        const samples: Split[][] = [[],[],[],[]];
        for (let round = 0; round < 4 + SAMPLE_BATCHES; round++) {
            for (let i = 0; i < 4; i++) {
                const index = (round+i)%4;
                const result = run(index<2,(index&1)!==0);
                if (round>=4) samples[index].push(result);
            }
        }
        // Components, like codecbench's vertex/index decoders, keep independent
        // minima. Their sum is not a measured full-model observation.
        const summary = samples.map(minima);
        const firstHalf = samples.map((samples) => minima(samples.slice(0, SAMPLE_BATCHES / 2)));
        for (const [name,index] of [["own",0],["reference",2]] as const) {
            const original = summary[index]; const plain = summary[index+1];
            console.log(JSON.stringify({ model, implementation: name, statistic: "independent minima", batches: SAMPLE_BATCHES, modelMs: original.total, plainModelMs: plain.total, modelFirstHalfMs: firstHalf[index].total, plainModelFirstHalfMs: firstHalf[index+1].total, setupMs: original.setup, teardownMs: original.teardown, boundaryCopiesMs: original.copies, etc1Ms: original.etc1, etc2RgbaMs: original["etc2-rgba"], astcWithZstdMs: original.astc, astcWithoutZstdMs: plain.astc, astcWithZstdFirstHalfMs: firstHalf[index].astc, astcWithoutZstdFirstHalfMs: firstHalf[index+1].astc, zstdDeltaMs: original.astc-plain.astc }));
        }
    }
}
