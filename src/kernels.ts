import { KernelSpec } from "./kernel";

export const registry: { [name: string]: KernelSpec } = {
    Add: {
        name: "Add",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "resultSize",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "firstArray",
                shaderType: "array<f32>",
            },
            {
                name: "secondArray",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "resultArray",
                shaderType: "array<f32>",
                size: "resultSize",
            },
        ],
        workgroupSize: [64, 1, 1],
        workgroupCount: ["resultSize/8", 1, 1],
        shader: `
    if (global_id.x >= parameters.resultSize) {
        return;
    }
    resultArray[global_id.x] = firstArray[global_id.x] + secondArray[global_id.x];
`
    },

    Add_: {
        name: "Add_",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "resultSize",
                shaderType: "u32",
            },
        ],
        inputs: [
            
            {
                name: "secondArray",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "firstArray",
                shaderType: "array<f32>",
                size: "resultSize",
            },
        ],
        workgroupSize: [64, 1, 1],
        workgroupCount: ["resultSize/8", 1, 1],
        shader: `
    if (global_id.x >= parameters.resultSize) {
        return;
    }
    firstArray[global_id.x] += secondArray[global_id.x];
`
    },

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
                size: "resultRows * resultCols",
            },
        ],
        workgroupSize: [8, 8, 1],
        workgroupCount: ["resultRows/8", "resultCols/8", 1],
        shader: `
    if (global_id.x >= parameters.resultRows || global_id.y >= u32(parameters.resultCols)) {
        return;
    }
    var result = 0.0;
    for (var i = 0u; i < parameters.innerDim; i = i + 1u) {
        let a = global_id.x * parameters.innerDim + i;
        let b = i * parameters.resultCols + global_id.y;
        result = result + firstMatrix[a] * secondMatrix[b];
    }
    let index = global_id.y + global_id.x * parameters.resultCols;
    resultMatrix[index] = result;
`
    },
};
