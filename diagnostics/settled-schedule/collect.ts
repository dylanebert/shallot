export {};

const port = Number(process.env.CDP_PORT ?? 9222);
const targetHint = process.argv[2] ?? "";
const targets = (await fetch(`http://127.0.0.1:${port}/json/list`).then((response) =>
    response.json(),
)) as Array<{ type: string; url: string; webSocketDebuggerUrl: string }>;
const target = targets.find(
    (candidate) =>
        candidate.type === "page" &&
        candidate.url.includes(targetHint) &&
        candidate.url.startsWith("http://"),
);
if (!target) throw new Error(`No headed Chrome page matched ${JSON.stringify(targetHint)}`);

const socket = new WebSocket(target.webSocketDebuggerUrl);
const requestId = 1;
const result = await new Promise<string>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Chrome DevTools request timed out")), 10_000);
    socket.addEventListener("open", () => {
        socket.send(
            JSON.stringify({
                id: requestId,
                method: "Runtime.evaluate",
                params: {
                    expression: "JSON.stringify(window.__shallotScheduleResult ?? null)",
                    returnByValue: true,
                },
            }),
        );
    });
    socket.addEventListener("message", (event) => {
        const message = JSON.parse(String(event.data)) as {
            id?: number;
            error?: { message: string };
            result?: { result?: { value?: string }; exceptionDetails?: { text: string } };
        };
        if (message.id !== requestId) return;
        clearTimeout(timeout);
        if (message.error) reject(new Error(message.error.message));
        else if (message.result?.exceptionDetails)
            reject(new Error(message.result.exceptionDetails.text));
        else if (message.result?.result?.value === undefined)
            reject(new Error("Chrome returned no schedule record"));
        else resolve(message.result.result.value);
        socket.close();
    });
    socket.addEventListener("error", () => {
        clearTimeout(timeout);
        reject(new Error("Chrome DevTools WebSocket failed"));
    });
});

const record = JSON.parse(result) as { complete?: boolean; stopReason?: string } | null;
if (!record) throw new Error("The selected page has not installed the schedule recorder");
console.log(JSON.stringify(record));
if (!record.complete) process.exitCode = 2;
if (record.stopReason && record.stopReason !== "30-second window complete") {
    console.error(`Stopped early: ${record.stopReason}`);
    process.exitCode = 3;
}
