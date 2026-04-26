export type Severity = "error" | "warning" | "info";

export interface LogEntry {
    severity: Severity;
    text: string;
    timestamp: number;
    count: number;
}

const MAX_ENTRIES = 500;
const TRIM_AMOUNT = 100;
const THROTTLE_MS = 1000;
const MAX_THROTTLE_KEYS = 200;

const _entries: LogEntry[] = [];
let _version = $state(0);
let _errorCount = 0;
let _warningCount = 0;
const _throttle = new Map<string, number>();

export function getEntries(): LogEntry[] {
    void _version;
    return _entries;
}

export function getLatest(): LogEntry | undefined {
    void _version;
    return _entries[_entries.length - 1];
}

export function getErrorCount(): number {
    void _version;
    return _errorCount;
}

export function getWarningCount(): number {
    void _version;
    return _warningCount;
}

export function push(severity: Severity, text: string) {
    const now = Date.now();
    const key = severity + "\0" + text;

    const lastSeen = _throttle.get(key);
    if (lastSeen !== undefined && now - lastSeen < THROTTLE_MS) return;
    if (_throttle.size >= MAX_THROTTLE_KEYS) _throttle.clear();
    _throttle.set(key, now);

    const last = _entries[_entries.length - 1];
    if (last && last.severity === severity && last.text === text) {
        last.count++;
        _version++;
        return;
    }

    _entries.push({ severity, text, timestamp: now, count: 1 });
    if (severity === "error") _errorCount++;
    else if (severity === "warning") _warningCount++;

    if (_entries.length > MAX_ENTRIES) {
        const removed = _entries.splice(0, TRIM_AMOUNT);
        for (const e of removed) {
            if (e.severity === "error") _errorCount--;
            else if (e.severity === "warning") _warningCount--;
        }
    }

    _version++;
}

export function clear() {
    _entries.length = 0;
    _errorCount = 0;
    _warningCount = 0;
    _throttle.clear();
    _version++;
}
