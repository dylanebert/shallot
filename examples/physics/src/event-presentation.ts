export interface EventMark {
    point: [number, number, number];
    end: [number, number, number];
    label: string;
}

export const EVENT_SEGMENT = { width: 2, depthBias: -1 };
export const EVENT_POINT = { width: 3, depthBias: -1 };

export function drawEventMark(
    mark: EventMark,
    line: (
        a: [number, number, number],
        b: [number, number, number],
        color: number,
        options: typeof EVENT_SEGMENT,
    ) => void,
): void {
    const p = mark.point;
    line(mark.end, p, 0xffb62e, EVENT_SEGMENT);
    line([p[0] - 0.2, p[1], p[2]], [p[0] + 0.2, p[1], p[2]], 0xffb62e, EVENT_POINT);
}
