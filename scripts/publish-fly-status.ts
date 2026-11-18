/** Latest started-event wall time from pinned flyctl 0.5.12's documented Machine status table. */
export function flyMachineStartedAt(status: string): number {
    const rows = status.split("\n").map((row) => row.split("│").map((cell) => cell.trim()));
    const at = Math.max(
        ...rows.filter((row) => row[0] === "started").map((row) => Date.parse(row[3] ?? "")),
    );
    if (!Number.isFinite(at))
        throw new Error("Fly Machine status started-event timestamps are missing or invalid.");
    return at;
}
