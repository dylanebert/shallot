export function formatBytes(bytes: number): string {
    if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    if (bytes >= 1024) return (bytes / 1024).toFixed(0) + " KB";
    return bytes.toFixed(0) + " B";
}

export const msFmt = (v: number): string => v.toFixed(2) + " ms";
