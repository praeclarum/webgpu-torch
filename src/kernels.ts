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
                shaderType: "f32",
            },
        ],
        inputs: [
            {
                name: "firstMatrix",
                shaderType: "array<f32>",
            },
            {
                name: "secondMatrix",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "resultMatrix",
                shaderType: "array<f32>",
            },
        ],
        shader: `
    if (global_id.x >= parameters.resultRows || global_id.y >= u32(parameters.resultCols)) {
        return;
    }
    var result = 0.0;
    for (var i = 0u; i < parameters.innerDim; i = i + 1u) {
        let a = global_id.x * parameters.innerDim + i;
        let b = i * parameters.resultCols + global_id.y;
        result = result + firstMatrix.numbers[a] * secondMatrix.numbers[b];
    }
    let index = global_id.y + global_id.x * parameters.resultCols;
    resultMatrix.numbers[index] = result;
`
    },
};
