import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { test } from "playwright/test";
import { sceneTests } from "../scenes";

test.use(pipelineWarnings);
sceneTests(["NormalTangentMirrorTest", "MetalRoughSpheres"]);
