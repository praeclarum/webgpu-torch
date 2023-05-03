import { KernelSpec } from "./kernel";

export const registry: KernelSpec[] = [
    {
        name: "mm",
        config: [],
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
];
