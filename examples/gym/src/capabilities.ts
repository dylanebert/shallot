export interface GymCapabilities {
    firstPerson: boolean;
    physics: boolean;
    audio: boolean;
    audioRoom: boolean;
    audioSources: boolean;
    dynamicCount: boolean;
    cameraMode: boolean;
    layout: boolean;
    renderTestShapes: boolean;
    renderTestVariants: boolean;
    physicsTestVariants: boolean;
    pileShapes: boolean;
    pointLights: boolean;
    renderTestText: boolean;
    renderTestArrow: boolean;
}

interface ScenarioDef {
    id: string | null;
    label: string;
    caps: GymCapabilities;
}

function caps(overrides: Partial<GymCapabilities>): GymCapabilities {
    return {
        firstPerson: false,
        physics: false,
        audio: false,
        audioRoom: false,
        audioSources: false,
        dynamicCount: false,
        cameraMode: false,
        layout: false,
        renderTestShapes: false,
        renderTestVariants: false,
        physicsTestVariants: false,
        pileShapes: false,
        pointLights: false,
        renderTestText: false,
        renderTestArrow: false,
        ...overrides,
    };
}

export const SCENARIOS: ScenarioDef[] = [
    {
        id: null,
        label: "benchmark",
        caps: caps({ dynamicCount: true, cameraMode: true, layout: true, pointLights: true }),
    },
    {
        id: "pile",
        label: "pile",
        caps: caps({ physics: true, dynamicCount: true, pileShapes: true }),
    },
    {
        id: "audio",
        label: "audio",
        caps: caps({
            firstPerson: true,
            physics: true,
            audio: true,
            audioRoom: true,
            audioSources: true,
        }),
    },
    {
        id: "render",
        label: "render",
        caps: caps({
            renderTestShapes: true,
            renderTestVariants: true,
            pointLights: true,
            renderTestText: true,
            renderTestArrow: true,
        }),
    },
    { id: "physics", label: "physics", caps: caps({ physics: true, physicsTestVariants: true }) },
    { id: "player", label: "player", caps: caps({ firstPerson: true, physics: true }) },
];

export function getCapabilities(scenarioName: string | null): GymCapabilities {
    const match = SCENARIOS.find((s) => s.id === scenarioName);
    if (match) return match.caps;
    return SCENARIOS.find((s) => s.id === "pile")!.caps;
}
