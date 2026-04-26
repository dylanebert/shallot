import type { State } from "@dylanebert/shallot";
import { DayNight } from "./lib";
import panelHtml from "./panel.html?raw";
import panelCss from "./panel.css?raw";

function formatTime(hour: number): string {
    const h = Math.floor(hour) % 24;
    const m = Math.floor((hour % 1) * 60);
    return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}`;
}

function timeLabel(hour: number): string {
    if (hour < 5) return "Night";
    if (hour < 7) return "Dawn";
    if (hour < 10) return "Morning";
    if (hour < 14) return "Noon";
    if (hour < 16) return "Afternoon";
    if (hour < 18) return "Golden Hour";
    if (hour < 19) return "Sunset";
    if (hour < 20) return "Twilight";
    if (hour < 21) return "Dusk";
    return "Night";
}

export function raytracingUI(container: HTMLElement, state: State): () => void {
    container.innerHTML = `<style>${panelCss}</style>${panelHtml}`;

    const slider = container.querySelector(".time-slider") as HTMLInputElement;
    const clock = container.querySelector(".time-clock") as HTMLElement;
    const label = container.querySelector(".time-label") as HTMLElement;

    function update(hour: number): void {
        const eid = state.only([DayNight]);
        if (eid >= 0) DayNight.hour[eid] = hour;
        clock.textContent = formatTime(hour);
        label.textContent = timeLabel(hour);
    }

    update(parseFloat(slider.value) / 10);

    slider.addEventListener("input", () => {
        update(parseFloat(slider.value) / 10);
    });

    return () => {};
}
