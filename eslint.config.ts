import parser from "@babel/eslint-parser";
import typegpu from "eslint-plugin-typegpu";

export default [
    { ignores: ["**/*.d.ts"] },
    {
        ...typegpu.configs.recommended,
        files: ["**/*.ts", "**/*.tsx"],
        languageOptions: {
            parser,
            parserOptions: {
                requireConfigFile: false,
                babelOptions: { plugins: ["@babel/plugin-syntax-typescript"] },
            },
        },
    },
];
