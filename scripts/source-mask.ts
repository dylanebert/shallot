// Regex-literal scanning, shared by the source maskers in `check-exports.ts` and
// `check-tumble-fp.ts`.
//
// Both scanners mask comments and string literals so a `//`, a quote or a `)` inside one cannot
// be read as code. Neither handled a **regex literal**, whose body may carry `/`, `"`, `'` or a
// backtick — so `/["']/` opened a string that swallowed the rest of the file and the scanner went
// blind past it while still reporting a clean pass. The engine twin (`maskTrivia`,
// `runtime/gpu-labels.test.ts`) was repaired first and witnessed red under a regex-literal
// fixture; this is that repair for `scripts/`, factored out rather than written twice.
//
// A regex literal cannot be recognized from the `/` alone — `a / b` is division. The
// discriminator is what precedes it: a regex may start where an *operand* may not follow, i.e.
// after punctuation/an operator, after one of the keywords below, or at the start of input.

// `}` is deliberately absent: a regex may follow a block's closing brace, but division after one
// (`${a}/b` in text a scanner preserved verbatim) is the commoner shape, and treating it as a
// regex start swallows the rest of the line. Matches the engine twin's set.
const REGEX_PRECEDING_PUNCTUATION = /[=(,:[!&|^~+\-*%<>?{;]/;

// Keywords after which a `/` opens a regex rather than dividing (`return /x/.test(s)`). The
// punctuation set alone misses these, and a missed regex is exactly the blind spot this file
// exists to close.
const REGEX_PRECEDING_KEYWORD =
    /\b(return|typeof|instanceof|in|of|case|do|else|yield|await|delete|void|new|throw)$/;

// True when a `/` at the current position opens a regex literal, given the code *already
// emitted* (masked) before it. Reading the masked tail rather than the raw source is what makes
// a `/` after a comment (`// note\n/re/`) resolve correctly — the comment is spaces by then.
export function isRegexLiteralStart(maskedTail: string): boolean {
    const trimmed = maskedTail.replace(/\s+$/, "");
    if (trimmed === "") return true;
    const prev = trimmed[trimmed.length - 1];
    const before = trimmed[trimmed.length - 2] ?? "";
    // Two punctuation members are ambiguous because they also *end* an operand: TypeScript's
    // non-null assertion (`counts.get(k)! / n`) and a postfix increment (`i++ / n`). In both the
    // `/` divides, so the operand-ending reading wins over the operator reading.
    if (prev === "!" && /[\w$)\]]/.test(before)) return false;
    if ((prev === "+" || prev === "-") && before === prev) return false;
    if (REGEX_PRECEDING_PUNCTUATION.test(prev)) return true;
    return REGEX_PRECEDING_KEYWORD.test(trimmed);
}

// Scan the regex literal opening at `start` (which must index a `/`). Returns the offset one past
// the literal — closing `/` plus flags — and whether the literal was terminated. A regex literal
// cannot span a newline, so an unterminated one ends at the line break and the caller reports it
// as an unparsed site rather than masking to end of file.
export function scanRegexLiteral(
    content: string,
    start: number,
): { end: number; terminated: boolean } {
    let i = start + 1;
    let inClass = false;
    while (i < content.length) {
        const c = content[i];
        if (c === "\n") return { end: i, terminated: false };
        if (c === "\\") {
            i += 2;
            continue;
        }
        if (c === "[") {
            inClass = true;
            i++;
            continue;
        }
        if (c === "]" && inClass) {
            inClass = false;
            i++;
            continue;
        }
        if (c === "/" && !inClass) {
            i++;
            while (i < content.length && /[dgimsuvy]/.test(content[i])) i++;
            return { end: i, terminated: true };
        }
        i++;
    }
    return { end: i, terminated: false };
}
