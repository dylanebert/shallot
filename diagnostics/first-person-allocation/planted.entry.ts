import { setFlagsFromString } from "node:v8";

export const control = () => ({ planted: true });

export default async function create(input: string) {
    let frame = 0;
    let sink = 0;
    const kept: object[] = [];
    function lateCompile(value: number) {
        return value * value + 1;
    }
    setFlagsFromString("--allow-natives-syntax");
    const prepare = new Function("fn", "%PrepareFunctionForOptimization(fn)");
    const optimize = new Function("fn", "%OptimizeFunctionOnNextCall(fn)");
    return {
        step() {
            frame++;
            if (input === "steady") kept.push({ frame });
            if (input === "compile" && frame === 121) {
                prepare(lateCompile);
                sink += lateCompile(1);
                sink += lateCompile(2);
                optimize(lateCompile);
                sink += lateCompile(3);
            }
        },
        dispose() {
            if (sink < 0) throw new Error("unreachable");
            kept.length = 0;
        },
    };
}
