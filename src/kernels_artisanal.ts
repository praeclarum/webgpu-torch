import { KernelSpec } from "./kernel";

export const kernels: { [name: string]: KernelSpec } = {
    mm: {
        name: "mm",
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
    if (global_id.x >= parameters.resultRows || global_id.y >= parameters.resultCols) {
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
    sum: {
        name: "sum",
        config: [
            {
                name: "dtype",
            },
            {
                name: "workgroupSize",
            },
        ],
        parameters: [
            {
                name: "size",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: ["workgroupSize", 1, 1],
        workgroupCount: [1, 1, 1],
        shader: `
    var local_sum = 0.0;
    // Load inputData into local memory
    for (var i = local_id.x; i < parameters.size; i += $$workgroupSize$$) {
        local_sum += input[i];
    }
    // Write partial group sum to outputData
    output[local_id.x] = local_sum;

    workgroupBarrier(); // Make sure all threads have completed summation

    // First thread sums up results from all other threads
    if (local_id.x == 0u) {
        for (var i = 1u; i < $$workgroupSize$$; i++) {
            local_sum += output[i];
        }
        // Store final sum in the first element of result array
        output[0] = local_sum;
    }
`
    },
};
