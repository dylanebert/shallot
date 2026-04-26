<script lang="ts">
    let {
        onspeedx,
        onspeedy,
        oncolor,
    }: {
        onspeedx: (v: number) => void;
        onspeedy: (v: number) => void;
        oncolor: (hex: string) => void;
    } = $props();

    let speedX = $state(60);
    let speedY = $state(30);
    let color = $state("#c77b58");

    function setSpeedX(v: number) {
        speedX = v;
        onspeedx(v);
    }

    function setSpeedY(v: number) {
        speedY = v;
        onspeedy(v);
    }

    function setColor(hex: string) {
        color = hex;
        oncolor(hex);
    }
</script>

<div class="panel">
    <div class="title">Svelte Cube</div>

    <label class="control">
        Speed X: {speedX.toFixed(0)}
        <input
            type="range"
            min={0}
            max={360}
            value={speedX}
            oninput={(e) => setSpeedX(Number(e.currentTarget.value))}
            class="slider"
        />
    </label>

    <label class="control">
        Speed Y: {speedY.toFixed(0)}
        <input
            type="range"
            min={0}
            max={360}
            value={speedY}
            oninput={(e) => setSpeedY(Number(e.currentTarget.value))}
            class="slider"
        />
    </label>

    <label class="control">
        Color
        <input
            type="color"
            value={color}
            oninput={(e) => setColor(e.currentTarget.value)}
            class="color-picker"
        />
    </label>
</div>

<style>
    .panel {
        pointer-events: auto;
        position: absolute;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 16px 24px;
        display: flex;
        gap: 16px;
        align-items: center;
        color: #e0e0e0;
        font-size: 13px;
        font-family: system-ui, sans-serif;
    }

    .title {
        font-weight: 600;
        font-size: 14px;
        white-space: nowrap;
    }

    .control {
        display: flex;
        flex-direction: column;
        gap: 4px;
        white-space: nowrap;
    }

    .slider {
        width: 120px;
        accent-color: #c77b58;
    }

    .color-picker {
        width: 40px;
        height: 28px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        background: transparent;
    }
</style>
