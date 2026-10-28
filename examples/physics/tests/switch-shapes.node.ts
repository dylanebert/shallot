import { setDefaultTimeout } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { testCategorySwitches } from "./switch-category";

setDefaultTimeout(TIMEOUT.node);

testCategorySwitches("Shapes");
