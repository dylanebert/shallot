import { afterAll, afterEach, beforeEach } from "bun:test";
import { inspect } from "node:util";

type ConsoleLevel = "warn" | "error";
type ConsoleCall = { level: ConsoleLevel; message: string };
type ExpectedCall = {
    level: ConsoleLevel;
    pattern: RegExp;
    observed: boolean;
    required: boolean;
};
type ExpectationOptions = { optional?: boolean };
type TestConsole = { calls: ConsoleCall[]; expected: ExpectedCall[] };

const original = {
    warn: console.warn.bind(console),
    error: console.error.bind(console),
};
let current: TestConsole | undefined;
const pendingCalls: ConsoleCall[] = [];
const pendingExpected: ExpectedCall[] = [];

function format(args: unknown[]): string {
    return args
        .map((arg) =>
            typeof arg === "string"
                ? arg
                : inspect(arg, { colors: false, depth: 4, breakLength: 120 }),
        )
        .join(" ");
}

function record(level: ConsoleLevel, args: unknown[]): void {
    (current?.calls ?? pendingCalls).push({ level, message: format(args) });
    original[level](...args);
}

const guarded = {
    warn: (...args: unknown[]) => record("warn", args),
    error: (...args: unknown[]) => record("error", args),
};

/** Declare one expected warning or error in the current test or its fixture setup. */
export function expectConsole(
    level: ConsoleLevel,
    pattern: RegExp,
    options: ExpectationOptions = {},
): void {
    (current?.expected ?? pendingExpected).push({
        level,
        pattern,
        observed: false,
        required: !options.optional,
    });
}

/** Declare one expected warning in the current test or its fixture setup. */
export function expectWarning(pattern: RegExp, options?: ExpectationOptions): void {
    expectConsole("warn", pattern, options);
}

/** Declare one expected error in the current test or its fixture setup. */
export function expectError(pattern: RegExp, options?: ExpectationOptions): void {
    expectConsole("error", pattern, options);
}

function failures(test: TestConsole, context: string): string[] {
    const unexpected = test.calls.filter((call) => {
        const expected = test.expected.find((item) => {
            if (item.observed || item.level !== call.level) return false;
            item.pattern.lastIndex = 0;
            return item.pattern.test(call.message);
        });
        if (expected) expected.observed = true;
        return expected === undefined;
    });
    const missing = test.expected.filter((item) => item.required && !item.observed);
    return [
        ...unexpected.map((call) => `Unexpected console.${call.level}${context}: ${call.message}`),
        ...missing.map(
            (item) => `Missing expected console.${item.level}${context} matching ${item.pattern}`,
        ),
    ];
}

export function installConsoleGuard(): void {
    console.warn = guarded.warn;
    console.error = guarded.error;

    beforeEach(() => {
        current = {
            calls: pendingCalls.splice(0),
            expected: pendingExpected.splice(0),
        };
        console.warn = guarded.warn;
        console.error = guarded.error;
    });

    afterEach(() => {
        const test = current;
        current = undefined;
        console.warn = guarded.warn;
        console.error = guarded.error;
        if (!test) return;
        const failed = failures(test, "");
        if (failed.length > 0) throw new Error(failed.join("\n"));
    });

    afterAll(() => {
        const test = {
            calls: pendingCalls.splice(0),
            expected: pendingExpected.splice(0),
        };
        const failed = failures(test, " outside a test");
        if (failed.length > 0) throw new Error(failed.join("\n"));
    });
}
