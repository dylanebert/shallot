<script lang="ts">
	let { icon, size = 14, strokeWidth = 1.5, class: className = "", style = "" }: {
		icon: string;
		size?: number;
		strokeWidth?: number;
		class?: string;
		style?: string;
	} = $props();
</script>

<span
	class="icon {className}"
	{style}
	style:--size="{size}px"
	style:--stroke-width={strokeWidth}
>{@html icon}</span>

<style>
	.icon {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		line-height: 0;
	}

	.icon :global(svg) {
		width: var(--size);
		height: var(--size);
		stroke-width: var(--stroke-width);
	}
</style>
