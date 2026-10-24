export interface CostRow {
    scene: string;
    quality: string;
    adapter: { vendor: string; architecture: string; device: string; description: string };
    browser: string;
    resolution: string;
    timestampQuery: string;
    samples: number;
    frameMs: number;
    spanSamples: Record<string, number>;
    spansMs: Record<string, number>;
}

export function costCells(row: CostRow): string[] {
    return [
        row.scene,
        row.quality,
        row.resolution,
        row.frameMs.toFixed(3),
        row.timestampQuery === "available"
            ? Object.entries(row.spansMs)
                  .map(([name, ms]) => `${name}: ${ms.toFixed(6)} ms`)
                  .join("\n") || "—"
            : row.timestampQuery,
    ];
}

function escapeHtml(value: string): string {
    return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

export function costReport(template: string, rows: readonly CostRow[]): string {
    const table = rows
        .map(
            (row) =>
                `<tr>${costCells(row)
                    .map((value) => `<td>${escapeHtml(value)}</td>`)
                    .join("")}</tr>`,
        )
        .join("\n");
    return template
        .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "")
        .replace('<p id="status">Running…</p>', '<p id="status">Complete — saved measurements</p>')
        .replace("<tbody></tbody>", `<tbody>${table}</tbody>`)
        .replace(
            '<pre id="results"></pre>',
            `<pre id="results">${escapeHtml(JSON.stringify(rows, null, 2))}</pre>`,
        );
}
