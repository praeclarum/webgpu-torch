import { KernelSpec } from "./kernel";

export const kernels: { [name: string]: KernelSpec } = {
    conv2d: {
        name: "conv2d",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: [
            {
                name: "batchSize",
                shaderType: "u32",
            },
            {
                name: "inputChannels",
                shaderType: "u32",
            },
            {
                name: "outputChannels",
                shaderType: "u32",
            },
            {
                name: "inputHeight",
                shaderType: "u32",
            },
            {
                name: "inputWidth",
                shaderType: "u32",
            },
            {
                name: "kernelHeight",
                shaderType: "u32",
            },
            {
                name: "kernelWidth",
                shaderType: "u32",
            },
            {
                name: "outputHeight",
                shaderType: "u32",
            },
            {
                name: "outputWidth",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>",
            },
            {
                name: "weight",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "batchSize * outputChannels * outputHeight * outputWidth",
            },
        ],
        workgroupSize: [4, 4, 1],
        workgroupCount: ["outputWidth/4", "outputHeight/4", 1],
        shader: `
    if (global_id.x >= parameters.outputWidth || global_id.y >= parameters.outputHeight) {
        return;
    }
    // input shape = [B, C, H, W]
    for (var batch = 0u; batch < parameters.batchSize; batch++) {
        for (var outputChannel = 0u; outputChannel < parameters.outputChannels; outputChannel++) {
            var result = 0.0;
            // Do the convolution
            for (var inputChannel = 0u; inputChannel < parameters.inputChannels; inputChannel++) {
                for (var kernelY = 0u; kernelY < parameters.kernelHeight; kernelY++) {
                    for (var kernelX = 0u; kernelX < parameters.kernelWidth; kernelX++) {
                        var inputY = global_id.y + kernelY;
                        var inputX = global_id.x + kernelX;
                        var inputIndex =
                            batch * parameters.inputChannels * parameters.inputHeight * parameters.inputWidth +
                            inputChannel * parameters.inputHeight * parameters.inputWidth +
                            inputY * parameters.inputWidth +
                            inputX;
                        var kernelIndex =
                            outputChannel * parameters.inputChannels * parameters.kernelHeight * parameters.kernelWidth +
                            inputChannel * parameters.kernelHeight * parameters.kernelWidth +
                            kernelY * parameters.kernelWidth +
                            kernelX;
                        result = result + input[inputIndex] * weight[kernelIndex];
                    }
                }
            }
            // Output
            let outputIndex = 
                batch * parameters.outputChannels * parameters.outputHeight * parameters.outputWidth +
                outputChannel * parameters.outputHeight * parameters.outputWidth +
                global_id.y * parameters.outputWidth +
                global_id.x;
            output[outputIndex] = result;
        }
    }
`
    },
    dot: {
        name: "dot",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "aRows",
                shaderType: "u32",
            },
            {
                name: "aRowStride",
                shaderType: "u32",
            },
            {
                name: "bRowStride",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>",
            },
            {
                name: "b",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "1",
            },
        ],
        workgroupSize: [1, 1, 1],
        workgroupCount: [1, 1, 1],
        shader: `
    var result = 0.0;
    var aIndex = 0u;
    var bIndex = 0u;
    for (var aRow = 0u; aRow < parameters.aRows; aRow = aRow + 1u) {
        result = result + a[aIndex] * b[bIndex];
        aIndex = aIndex + parameters.aRowStride;
        bIndex = bIndex + parameters.bRowStride;
    }
    output[0] = result;
`
    },
    mv: {
        name: "mv",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "aRows",
                shaderType: "u32",
            },
            {
                name: "aCols",
                shaderType: "u32",
            },
            {
                name: "aRowStride",
                shaderType: "u32",
            },
            {
                name: "aColStride",
                shaderType: "u32",
            },
            {
                name: "bRowStride",
                shaderType: "u32",
            },
       ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>",
            },
            {
                name: "b",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "aRows",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["aRows/256", 1, 1],
        shader: `
    let outputRow = global_id.x;
    if (outputRow >= parameters.aRows) {
        return;
    }
    var result = 0.0;
    var aIndex = outputRow * parameters.aRowStride;
    var bIndex = 0u;
    for (var aCol = 0u; aCol < parameters.aCols; aCol = aCol + 1u) {
        result = result + a[aIndex] * b[bIndex];
        aIndex = aIndex + parameters.aColStride;
        bIndex = bIndex + parameters.bRowStride;
    }
    output[outputRow] = result;
`
    },
    mm: {
        name: "mm",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "aRows",
                shaderType: "u32",
            },
            {
                name: "aCols",
                shaderType: "u32",
            },
            {
                name: "bCols",
                shaderType: "u32",
            },
            {
                name: "aRowStride",
                shaderType: "u32",
            },
            {
                name: "aColStride",
                shaderType: "u32",
            },
            {
                name: "bRowStride",
                shaderType: "u32",
            },
            {
                name: "bColStride",
                shaderType: "u32",
            },
            {
                name: "alpha",
                shaderType: "f32",
            },
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>",
            },
            {
                name: "b",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "aRows * bCols",
            },
        ],
        workgroupSize: [16, 16, 1],
        workgroupCount: ["aRows/16", "bCols/16", 1],
        shader: `
    let outputRow = global_id.x;
    let outputCol = global_id.y;
    if (outputRow >= parameters.aRows || outputCol >= parameters.bCols) {
        return;
    }
    var result = 0.0;
    var aIndex = outputRow * parameters.aRowStride;
    var bIndex = outputCol * parameters.bColStride;
    for (var aCol = 0u; aCol < parameters.aCols; aCol = aCol + 1u) {
        result = result + a[aIndex] * b[bIndex];
        aIndex = aIndex + parameters.aColStride;
        bIndex = bIndex + parameters.bRowStride;
    }
    let outputIndex = outputCol + outputRow * parameters.bCols;
    output[outputIndex] = result;
`
    },
    bmm: {
        name: "bmm",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "batchSize",
                shaderType: "u32",
            },
            {
                name: "aRows",
                shaderType: "u32",
            },
            {
                name: "aCols",
                shaderType: "u32",
            },
            {
                name: "bCols",
                shaderType: "u32",
            },
            {
                name: "aBatchStride",
                shaderType: "u32",
            },
            {
                name: "aRowStride",
                shaderType: "u32",
            },
            {
                name: "aColStride",
                shaderType: "u32",
            },
            {
                name: "bBatchStride",
                shaderType: "u32",
            },
            {
                name: "bRowStride",
                shaderType: "u32",
            },
            {
                name: "bColStride",
                shaderType: "u32",
            },
            {
                name: "alpha",
                shaderType: "f32",
            },
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>",
            },
            {
                name: "b",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "batchSize * aRows * bCols",
            },
        ],
        workgroupSize: [8, 8, 4],
        workgroupCount: ["aRows/8", "bCols/8", "batchSize/4"],
        shader: `
    let outputRow = global_id.x;
    let outputCol = global_id.y;
    let outputBatch = global_id.z;
    if (outputRow >= parameters.aRows || outputCol >= parameters.bCols || outputBatch >= parameters.batchSize) {
        return;
    }
    var result = 0.0;
    var aIndex = outputBatch * parameters.aBatchStride + outputRow * parameters.aRowStride;
    var bIndex = outputBatch * parameters.bBatchStride + outputCol * parameters.bColStride;
    for (var aCol = 0u; aCol < parameters.aCols; aCol = aCol + 1u) {
        result = result + a[aIndex] * b[bIndex];
        aIndex = aIndex + parameters.aColStride;
        bIndex = bIndex + parameters.bRowStride;
    }
    let outputRowStride = parameters.bCols;
    let outputBatchStride = parameters.aRows * outputRowStride;
    let outputIndex = outputBatch * outputBatchStride + outputRow * outputRowStride + outputCol;
    output[outputIndex] = result;
`
    },
    sumDim: {
        name: "sumDim",
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
                name: "dimToSum",
                shaderType: "u32",
            },
            {
                name: "inputShape",
                shaderType: "vec3<u32>",
            },
            {
                name: "inputStrides",
                shaderType: "vec3<u32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: ["workgroupSize", "workgroupSize", "workgroupSize"],
        workgroupCount: [1, 1, 1],
        shader: `
        // Global index flattening for the reformatted 3D tensor
        var flatGlobalId: u32 = global_id.x * parameters.inputStrides.x + global_id.y * parameters.inputStrides.y + global_id.z * parameters.inputStrides.z;
    
        // Initialize sum
        var sum: f32 = 0.0;
    
        let numReductions: u32 = parameters.inputShape.y;
    
        // Sum reduction
        for (var i: u32 = 0; i < numReductions; i = i + 1) {
            // Compute the input index by adding the reduction offset to the current flat global index
            var dataIndex: u32 = flatGlobalId + i * parameters.inputStrides.y;
    
            if (dataIndex < input.length()) {
                // Accumulate the input value into sum
                sum = sum + input[dataIndex];
            }
        }
    
        // Write the reduced sum value to output tensor
        if (flatGlobalId < output.length()) {
            output[flatGlobalId] = sum;
        }
    `
    },
    uniform_: {
        name: "uniform_",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: [
            {
                name: "size",
                shaderType: "u32",
            },
            {
                name: "seed",
                shaderType: "u32",
            },
            {
                name: "lowerBound",
                shaderType: "f32",
            },
            {
                name: "upperBound",
                shaderType: "f32",
            },
        ],
        inputs: [],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "size",
            }
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["size/256", 1, 1],
        shader: `
        let outputIndex = global_id.x;
        if (outputIndex >= parameters.size) {
            return;
        }

        var b = 0u;
        let seed = u32(u32(parameters.seed + outputIndex) * 1099087573u);
        b = ((seed << 13) ^ seed) >> 19;
        let z1 = ((seed & 429496729u) << 12) ^ b;
        b = ((seed << 2) ^ seed) >> 25;
        let z2 = ((seed & 4294967288u) << 4) ^ b;
        b = ((seed << 3) ^ seed) >> 11;
        let z3 = ((seed & 429496280u) << 17) ^ b;
        let z4 = u32(u32(1664525u * seed) + 1013904223u);
        let r = z1 ^ z2 ^ z3 ^ z4;
        let u = f32(u32(r)) * 2.3283064365387e-10;

        output[outputIndex] = parameters.lowerBound + u * (parameters.upperBound - parameters.lowerBound);
    `
    }
};
