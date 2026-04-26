<script lang="ts">
	import { getEntries, clear, type Severity } from "./log.svelte.js";

	let { onclose }: { onclose: () => void } = $props();

	let filter: Severity | null = $state(null);
	let body: HTMLElement;
	let height = $state(200);

	let filtered = $derived.by(() => {
		const entries = getEntries();
		return entries.filter((e) => !filter || e.severity === filter);
	});

	function shouldAutoScroll() {
		if (!body) return true;
		return body.scrollHeight - body.scrollTop - body.clientHeight < 40;
	}

	let wasAtBottom = true;

	$effect(() => {
		void filtered.length;
		if (wasAtBottom && body) {
			requestAnimationFrame(() => {
				body.scrollTop = body.scrollHeight;
			});
		}
	});

	function onscroll() {
		wasAtBottom = shouldAutoScroll();
	}

	function formatTime(ts: number) {
		const d = new Date(ts);
		return d.toTimeString().slice(0, 8);
	}

	const filters: { label: string; value: Severity | null }[] = [
		{ label: "All", value: null },
		{ label: "Errors", value: "error" },
		{ label: "Warnings", value: "warning" },
		{ label: "Info", value: "info" },
	];

	let dragging = $state(false);
	let startY = 0;
	let startHeight = 0;

	function onresizestart(e: PointerEvent) {
		dragging = true;
		startY = e.clientY;
		startHeight = height;
		(e.target as HTMLElement).setPointerCapture(e.pointerId);
	}

	function onresizemove(e: PointerEvent) {
		if (!dragging) return;
		const delta = startY - e.clientY;
		height = Math.max(80, Math.min(startHeight + delta, window.innerHeight * 0.7));
	}

	function onresizeend() {
		dragging = false;
	}
</script>

<div class="drawer" style:height="{height}px">
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="resize-handle"
		class:active={dragging}
		onpointerdown={onresizestart}
		onpointermove={onresizemove}
		onpointerup={onresizeend}
	></div>
	<div class="header">
		<div class="filters">
			{#each filters as f}
				<button
					class="filter-btn"
					class:active={filter === f.value}
					onclick={() => filter = f.value}
				>{f.label}</button>
			{/each}
		</div>
		<div class="actions">
			<button class="action-btn" onclick={clear} title="Clear">
				<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
					<path d="M3 4h10M6 4V3h4v1M5 4v9h6V4" />
					<path d="M7 7v4M9 7v4" />
				</svg>
			</button>
			<button class="action-btn close-btn" onclick={onclose} title="Close">
				<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
					<path d="M4 4l4 4 4-4" />
				</svg>
			</button>
		</div>
	</div>
	<div class="body" bind:this={body} {onscroll}>
		{#each filtered as entry}
			<div class="row" class:row-error={entry.severity === "error"} class:row-warning={entry.severity === "warning"}>
				<span class="time">{formatTime(entry.timestamp)}</span>
				<span class="indicator" class:error={entry.severity === "error"} class:warning={entry.severity === "warning"} class:info={entry.severity === "info"}></span>
				<span class="text">{entry.text}</span>
				{#if entry.count > 1}
					<span class="count">{entry.count}</span>
				{/if}
			</div>
		{/each}
	</div>
</div>

<style>
	.drawer {
		display: flex;
		flex-direction: column;
		background: var(--bg);
		flex-shrink: 0;
	}

	.resize-handle {
		height: 1px;
		background: var(--border);
		cursor: row-resize;
		flex-shrink: 0;
		position: relative;
		z-index: 2;
		transition: background 150ms var(--ease-out);
	}

	.resize-handle::before {
		content: "";
		position: absolute;
		left: 0;
		right: 0;
		top: -3px;
		bottom: -3px;
	}

	.resize-handle:hover,
	.resize-handle.active {
		background: var(--accent);
	}

	.header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		height: 28px;
		padding: 0 8px;
		background: var(--surface-1);
		flex-shrink: 0;
	}

	.filters {
		display: flex;
		gap: 2px;
	}

	.filter-btn {
		padding: 2px 8px;
		border: none;
		border-radius: 3px;
		background: transparent;
		color: var(--text-muted);
		font-size: 11px;
		cursor: pointer;
		transition: all 120ms var(--ease-out);
	}

	.filter-btn:hover {
		color: var(--text-secondary);
		background: var(--surface-2);
	}

	.filter-btn.active {
		color: var(--accent);
		background: rgba(212, 149, 96, 0.1);
	}

	.actions {
		display: flex;
		gap: 2px;
	}

	.action-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 22px;
		height: 22px;
		border: none;
		border-radius: 3px;
		background: transparent;
		color: var(--text-muted);
		cursor: pointer;
		padding: 0;
		transition: all 120ms var(--ease-out);
	}

	.action-btn:hover {
		color: var(--text);
		background: var(--surface-2);
	}

	.action-btn svg {
		width: 12px;
		height: 12px;
	}

	.close-btn svg {
		width: 10px;
		height: 10px;
	}

	.body {
		flex: 1;
		overflow-y: auto;
		font-family: ui-monospace, "SF Mono", "Cascadia Code", monospace;
		font-size: 11px;
		scrollbar-gutter: stable;
	}

	.body::-webkit-scrollbar { width: 6px; }
	.body::-webkit-scrollbar-track { background: transparent; }
	.body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

	.row {
		display: flex;
		align-items: baseline;
		gap: 8px;
		padding: 1px 10px;
		min-height: 20px;
		border-left: 2px solid transparent;
	}

	.row:hover {
		background: var(--surface-1);
	}

	.row-error {
		border-left-color: #e85050;
		background: rgba(232, 80, 80, 0.03);
	}

	.row-warning {
		border-left-color: #e8a520;
		background: rgba(232, 165, 32, 0.03);
	}

	.time {
		color: var(--text-muted);
		flex-shrink: 0;
		opacity: 0.5;
		font-size: 10px;
	}

	.indicator {
		width: 5px;
		height: 5px;
		border-radius: 50%;
		flex-shrink: 0;
		position: relative;
		top: -1px;
	}

	.indicator.error { background: #e85050; }
	.indicator.warning { background: #e8a520; }
	.indicator.info { background: var(--text-muted); opacity: 0.5; }

	.text {
		flex: 1;
		color: var(--text-secondary);
		white-space: pre-wrap;
		word-break: break-word;
	}

	.count {
		flex-shrink: 0;
		padding: 0 5px;
		border-radius: 8px;
		background: var(--surface-3);
		color: var(--text-muted);
		font-size: 10px;
		line-height: 16px;
	}
</style>
