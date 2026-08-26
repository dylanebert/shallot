/** hysteresis row reorder: nudge each row toward its ranked position only when
 *  it's displaced by ≥2 slots (avoids 1-slot jitter). Pure — takes the current
 *  display order and a map of name→ranked-index, returns the new order. */
export function reorderRows(order: string[], rank: Map<string, number>): string[] {
    const next = order.slice();
    for (let i = 0; i < next.length; i++) {
        const desired = rank.get(next[i])!;
        if (Math.abs(desired - i) >= 2) {
            const [moved] = next.splice(i, 1);
            next.splice(desired, 0, moved);
        }
    }
    return next;
}
