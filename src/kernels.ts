import { KernelSpec } from "./kernel";

export const registry: { [name: string]: KernelSpec } = {
    MM: {
        name: "MM",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "resultRows",
                shaderType: "u32",
            },
            {
                name: "resultCols",
                shaderType: "u32",
            },
            {
                name: "innerDim",
                shaderType: "u32",
            },
            {
                name: "alpha",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "firstMatrix",
                shaderType: "f32",
            },
            {
                name: "secondMatrix",
                shaderType: "f32",
            },
        ],
        outputs: [
            {
                name: "resultMatrix",
                shaderType: "f32",
            },
        ],
        shader: "",
    },
};
