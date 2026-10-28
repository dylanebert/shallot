import { setDefaultTimeout } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { testDominoSwitches } from "./switch-dominoes";

setDefaultTimeout(TIMEOUT.node);

testDominoSwitches("first");
