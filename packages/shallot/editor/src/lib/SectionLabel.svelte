<script lang="ts">
    let { label }: { label: string } = $props();
</script>

<div class="panel-section-label">{label}</div>

<style>
    .panel-section-label {
        padding: 8px 12px 4px;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
    }
</style>
