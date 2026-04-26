<script lang="ts">
    import type { SlotConfig, ModRoute } from "./graph";
    import { getDisplaySlots } from "./graph";

    let {
        slots,
        modRoutes,
    }: {
        slots: SlotConfig[];
        modRoutes: ModRoute[];
    } = $props();

    const NODE_W = 88;
    const NODE_H = 32;
    const GAP_X = 24;
    const GAP_Y = 16;
    const PAD = 16;

    const COLORS: Record<string, string> = {
        oscillator: "#d49560",
        sample: "#c97060",
        filter: "#60b0d4",
        envelope: "#60d480",
        constant: "#d460b0",
        gain: "#d4d460",
        mix: "#9060d4",
    };

    interface NodePos {
        id: string;
        type: string;
        label: string;
        x: number;
        y: number;
    }

    interface Edge {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
        dashed: boolean;
    }

    let layout = $derived.by(() => {
        const display = getDisplaySlots(slots);
        const sources = display.filter((s) => s.type === "oscillator" || s.type === "sample");
        const mixNodes = display.filter((s) => s.type === "mix");
        const filters = display.filter((s) => s.type === "filter");
        const ampEnv = display.filter((s) => s.id === "ampEnv");
        const vol = display.filter((s) => s.id === "vol");
        const modNodes = display.filter(
            (s) => (s.type === "envelope" && s.id !== "ampEnv") || s.type === "constant",
        );

        const hasMix = mixNodes.length > 0;
        const postMerge = [...filters, ...ampEnv, ...vol];

        const nodes: NodePos[] = [];
        const positions = new Map<string, { x: number; y: number }>();
        const edges: Edge[] = [];

        let cx = PAD;

        if (hasMix && sources.length > 1) {
            if (sources.length === 2) {
                const oscStackH = 2 * NODE_H + GAP_Y;
                for (let i = 0; i < 2; i++) {
                    const slot = sources[i];
                    const y = PAD + i * (NODE_H + GAP_Y);
                    nodes.push({ id: slot.id, type: slot.type, label: slot.label, x: cx, y });
                    positions.set(slot.id, { x: cx, y });
                }
                cx += NODE_W + GAP_X;

                const mixSlot = mixNodes[0];
                const mixY = PAD + (oscStackH - NODE_H) / 2;
                nodes.push({ id: mixSlot.id, type: mixSlot.type, label: mixSlot.label, x: cx, y: mixY });
                positions.set(mixSlot.id, { x: cx, y: mixY });
                for (const osc of sources) {
                    const op = positions.get(osc.id)!;
                    edges.push({
                        x1: op.x + NODE_W,
                        y1: op.y + NODE_H / 2,
                        x2: cx,
                        y2: mixY + NODE_H / 2,
                        dashed: false,
                    });
                }
                cx += NODE_W + GAP_X;

                let prev = positions.get(mixSlot.id)!;
                for (const slot of postMerge) {
                    const y = mixY;
                    nodes.push({ id: slot.id, type: slot.type, label: slot.label, x: cx, y });
                    positions.set(slot.id, { x: cx, y });
                    edges.push({
                        x1: prev.x + NODE_W,
                        y1: prev.y + NODE_H / 2,
                        x2: cx,
                        y2: y + NODE_H / 2,
                        dashed: false,
                    });
                    prev = { x: cx, y };
                    cx += NODE_W + GAP_X;
                }
            } else {
                const osc1Y = PAD;
                const osc2Y = PAD + NODE_H + GAP_Y;
                const osc3Y = PAD + 2 * (NODE_H + GAP_Y);
                nodes.push({ id: sources[0].id, type: sources[0].type, label: sources[0].label, x: cx, y: osc1Y });
                nodes.push({ id: sources[1].id, type: sources[1].type, label: sources[1].label, x: cx, y: osc2Y });
                positions.set(sources[0].id, { x: cx, y: osc1Y });
                positions.set(sources[1].id, { x: cx, y: osc2Y });
                const osc3X = cx;
                cx += NODE_W + GAP_X;

                const mix1 = mixNodes[0];
                const mix1Y = PAD + (NODE_H + GAP_Y) / 2;
                nodes.push({ id: mix1.id, type: mix1.type, label: mix1.label, x: cx, y: mix1Y });
                positions.set(mix1.id, { x: cx, y: mix1Y });
                edges.push({ x1: osc3X + NODE_W, y1: osc1Y + NODE_H / 2, x2: cx, y2: mix1Y + NODE_H / 2, dashed: false });
                edges.push({ x1: osc3X + NODE_W, y1: osc2Y + NODE_H / 2, x2: cx, y2: mix1Y + NODE_H / 2, dashed: false });

                nodes.push({ id: sources[2].id, type: sources[2].type, label: sources[2].label, x: cx, y: osc3Y });
                positions.set(sources[2].id, { x: cx, y: osc3Y });
                cx += NODE_W + GAP_X;

                const mix2 = mixNodes[1];
                const mix2Y = PAD + NODE_H + GAP_Y;
                nodes.push({ id: mix2.id, type: mix2.type, label: mix2.label, x: cx, y: mix2Y });
                positions.set(mix2.id, { x: cx, y: mix2Y });
                edges.push({ x1: positions.get(mix1.id)!.x + NODE_W, y1: mix1Y + NODE_H / 2, x2: cx, y2: mix2Y + NODE_H / 2, dashed: false });
                edges.push({ x1: positions.get(sources[2].id)!.x + NODE_W, y1: osc3Y + NODE_H / 2, x2: cx, y2: mix2Y + NODE_H / 2, dashed: false });
                cx += NODE_W + GAP_X;

                let prev = positions.get(mix2.id)!;
                for (const slot of postMerge) {
                    const y = mix2Y;
                    nodes.push({ id: slot.id, type: slot.type, label: slot.label, x: cx, y });
                    positions.set(slot.id, { x: cx, y });
                    edges.push({
                        x1: prev.x + NODE_W,
                        y1: prev.y + NODE_H / 2,
                        x2: cx,
                        y2: y + NODE_H / 2,
                        dashed: false,
                    });
                    prev = { x: cx, y };
                    cx += NODE_W + GAP_X;
                }
            }
        } else {
            const chain = [...sources, ...mixNodes, ...postMerge];
            for (let i = 0; i < chain.length; i++) {
                const slot = chain[i];
                const y = PAD;
                nodes.push({ id: slot.id, type: slot.type, label: slot.label, x: cx, y });
                positions.set(slot.id, { x: cx, y });
                if (i > 0) {
                    const from = positions.get(chain[i - 1].id)!;
                    edges.push({
                        x1: from.x + NODE_W,
                        y1: from.y + NODE_H / 2,
                        x2: cx,
                        y2: y + NODE_H / 2,
                        dashed: false,
                    });
                }
                cx += NODE_W + GAP_X;
            }
        }

        const mainBottom = Math.max(...nodes.map((n) => n.y + NODE_H));
        let my = mainBottom + GAP_Y;
        for (const slot of modNodes) {
            const mx = PAD;
            nodes.push({ id: slot.id, type: slot.type, label: slot.label, x: mx, y: my });
            positions.set(slot.id, { x: mx, y: my });
            my += NODE_H + GAP_Y;
        }

        for (const route of modRoutes) {
            const src = positions.get(route.source);
            const tgt = positions.get(route.target);
            if (!src || !tgt) continue;
            edges.push({
                x1: src.x + NODE_W,
                y1: src.y + NODE_H / 2,
                x2: tgt.x + NODE_W / 2,
                y2: tgt.y + NODE_H,
                dashed: true,
            });
        }

        const w = cx + PAD;
        const h = Math.max(NODE_H + PAD * 2, my + PAD);
        return { nodes, edges, w, h };
    });
</script>

<div class="signal-flow">
    <span class="section-label">Signal Flow</span>
    <svg viewBox="0 0 {layout.w} {layout.h}" xmlns="http://www.w3.org/2000/svg">
        {#each layout.edges as edge}
            <line
                x1={edge.x1}
                y1={edge.y1}
                x2={edge.x2}
                y2={edge.y2}
                stroke={edge.dashed ? "var(--text-muted)" : "var(--text-secondary)"}
                stroke-width="1"
                stroke-dasharray={edge.dashed ? "4 3" : "none"}
            />
        {/each}
        {#each layout.nodes as node}
            <rect
                x={node.x}
                y={node.y}
                width={NODE_W}
                height={NODE_H}
                rx="4"
                fill="var(--surface-2)"
                stroke={COLORS[node.type] ?? "var(--border)"}
                stroke-width="1.5"
            />
            <text
                x={node.x + NODE_W / 2}
                y={node.y + NODE_H / 2}
                text-anchor="middle"
                dominant-baseline="central"
                fill="var(--text-secondary)"
                font-size="11"
                font-family="JetBrains Mono, monospace"
            >{node.label}</text>
        {/each}
    </svg>
</div>

<style>
    .signal-flow {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    svg {
        width: 100%;
        min-height: 120px;
        border: 1px solid var(--border);
        border-radius: 4px;
        background: var(--surface-1);
    }
</style>
