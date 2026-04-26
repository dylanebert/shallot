import { useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { Part, type State } from "@dylanebert/shallot";
import { RotateCube } from "./lib";

function Panel({ state }: { state: State }) {
    const [speedX, setSpeedX] = useState(60);
    const [speedY, setSpeedY] = useState(30);
    const [color, setColor] = useState("#c77b58");

    useEffect(() => {
        const eid = state.only([RotateCube]);
        if (eid >= 0) {
            setSpeedX(RotateCube.speedX[eid]);
            setSpeedY(RotateCube.speedY[eid]);
        }
    }, []);

    function updateSpeed(axis: "x" | "y", value: number) {
        const eid = state.only([RotateCube]);
        if (eid < 0) return;
        if (axis === "x") {
            RotateCube.speedX[eid] = value;
            setSpeedX(value);
        } else {
            RotateCube.speedY[eid] = value;
            setSpeedY(value);
        }
    }

    function updateColor(hex: string) {
        setColor(hex);
        const eid = state.only([RotateCube]);
        if (eid >= 0) Part.color[eid] = parseInt(hex.slice(1), 16);
    }

    return (
        <div style={panelStyle}>
            <div style={titleStyle}>React Cube</div>

            <label style={labelStyle}>
                Speed X: {speedX.toFixed(0)}
                <input
                    type="range"
                    min={0}
                    max={360}
                    value={speedX}
                    onChange={(e) => updateSpeed("x", Number(e.target.value))}
                    style={sliderStyle}
                />
            </label>

            <label style={labelStyle}>
                Speed Y: {speedY.toFixed(0)}
                <input
                    type="range"
                    min={0}
                    max={360}
                    value={speedY}
                    onChange={(e) => updateSpeed("y", Number(e.target.value))}
                    style={sliderStyle}
                />
            </label>

            <label style={labelStyle}>
                Color
                <input
                    type="color"
                    value={color}
                    onChange={(e) => updateColor(e.target.value)}
                    style={colorStyle}
                />
            </label>
        </div>
    );
}

export function reactUI(container: HTMLElement, state: State): () => void {
    const root = createRoot(container);
    root.render(<Panel state={state} />);
    return () => root.unmount();
}

const panelStyle: React.CSSProperties = {
    pointerEvents: "auto",
    position: "absolute",
    bottom: 24,
    left: "50%",
    transform: "translateX(-50%)",
    background: "rgba(0, 0, 0, 0.7)",
    backdropFilter: "blur(12px)",
    borderRadius: 12,
    padding: "16px 24px",
    display: "flex",
    gap: 16,
    alignItems: "center",
    color: "#e0e0e0",
    fontSize: 13,
    fontFamily: "system-ui, sans-serif",
};

const titleStyle: React.CSSProperties = {
    fontWeight: 600,
    fontSize: 14,
    whiteSpace: "nowrap",
};

const labelStyle: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    whiteSpace: "nowrap",
};

const sliderStyle: React.CSSProperties = {
    width: 120,
    accentColor: "#c77b58",
};

const colorStyle: React.CSSProperties = {
    width: 40,
    height: 28,
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    background: "transparent",
};
