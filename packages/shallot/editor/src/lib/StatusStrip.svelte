<script lang="ts">
	import { getLatest, getErrorCount, getWarningCount } from "./log.svelte.js";

	let { onopen }: { onopen: () => void } = $props();

	let latest = $derived(getLatest());
	let errorCount = $derived(getErrorCount());
	let warningCount = $derived(getWarningCount());
</script>

<button class="strip" onclick={onopen}>
	{#if errorCount > 0}
		<span class="badge error"><span class="dot"></span>{errorCount}</span>
	{/if}
	{#if warningCount > 0}
		<span class="badge warning"><span class="dot"></span>{warningCount}</span>
	{/if}
	{#if latest}
		{#key latest}
			<span class="message" class:error={latest.severity === "error"} class:warning={latest.severity === "warning"}>{latest.text}</span>
		{/key}
	{/if}
</button>

<style>
	.strip {
		display: flex;
		align-items: center;
		gap: 10px;
		height: 26px;
		padding: 0 10px;
		background: var(--bg);
		border-top: 1px solid var(--border);
		cursor: pointer;
		font-size: 11px;
		color: var(--text-muted);
		width: 100%;
		transition: background 120ms var(--ease-out);
	}

	.strip:hover {
		background: var(--surface-1);
	}

	.badge {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		font-variant-numeric: tabular-nums;
	}

	.badge.error { color: #e85050; }
	.badge.warning { color: #e8a520; }

	.dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: currentColor;
	}

	.message {
		font-family: ui-monospace, "SF Mono", "Cascadia Code", monospace;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		min-width: 0;
		animation: fade-in 160ms var(--ease-out);
	}

	.message.error { color: #e85050; }
	.message.warning { color: #e8a520; }

	@keyframes fade-in {
		from { opacity: 0; transform: translateY(2px); }
		to { opacity: 1; transform: translateY(0); }
	}
</style>
