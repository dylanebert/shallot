/** One line per solve: 16 hex digits, optionally a space and six event lists. Omitted lists are empty. */
export function sampleGold(text: string): { hash: bigint; events: (number | string)[][][] }[] {
    return text
        .trimEnd()
        .split("\n")
        .map((line, step) => {
            const match = /^([0-9a-f]{16})(?: (.+))?$/.exec(line);
            if (!match) throw new Error(`step ${step}: invalid sample gold line`);
            let events: unknown;
            try {
                events = match[2] ? JSON.parse(match[2]) : [[], [], [], [], [], []];
            } catch (cause) {
                throw new Error(`step ${step}: invalid sample event records`, { cause });
            }
            if (
                !Array.isArray(events) ||
                events.length !== 6 ||
                !events.every(
                    (channel) =>
                        Array.isArray(channel) &&
                        channel.every(
                            (record) =>
                                Array.isArray(record) &&
                                record.every(
                                    (value) =>
                                        typeof value === "number" || typeof value === "string",
                                ),
                        ),
                )
            )
                throw new Error(`step ${step}: invalid sample event records`);
            return { hash: BigInt(`0x${match[1]}`), events };
        });
}
