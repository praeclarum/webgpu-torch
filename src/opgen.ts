// Generate code from op_spec.ts and op_table.ts
import { KernelInputSpec, KernelOutputSpec, KernelParamSpec, KernelSpec } from "./kernel";
import {
    ExprCode,
    exprCodeToWebGLShader,
    exprNodeToWebGLShader,
    parseCode,
    substituteIdentifiers,
} from "./expr";
import { BinaryOpSpec, OpSpec, ReductionOpSpec, UnaryOpSpec } from "./op_spec";

export class CodeWriter {
    private indentLevel = 0;
    private lines: string[] = [];
    indent() {
        this.indentLevel++;
    }
    dedent() {
        this.indentLevel--;
    }
    writeLine(line: string) {
        this.lines.push("    ".repeat(this.indentLevel) + line);
    }
    toString() {
        return this.lines.join("\n");
    }
}

export function opSpecToKernelSpecs(op: OpSpec): KernelSpec[] {
    if (op.type == "reduction") {
        return getReductionKernelSpecs(op as ReductionOpSpec);
    } else if (op.type == "binary") {
        return getBinaryKernelSpecs(op as BinaryOpSpec);
    } else {
        return getUnaryKernelSpecs(op as UnaryOpSpec);
    }
}

function getReductionKernelSpecs(op: ReductionOpSpec): KernelSpec[] {
    const specs = [getReductionKernelSpec(op), getReductionDimKernelSpec(op)];
    if (op.backward) {
        specs.push(getReductionGradKernelSpec(op, op.backward));
    }
    return specs;
}

function getBinaryKernelSpecs(op: BinaryOpSpec): KernelSpec[] {
    const specs = [
        getBinaryKernelSpec(op, false, false, false),
        getBinaryKernelSpec(op, true, false, false),
        getBinaryKernelSpec(op, false, true, false),
        getBinaryKernelSpec(op, true, true, false),
        getBinaryKernelSpec(op, false, false, true),
        getBinaryKernelSpec(op, true, false, true),
    ];
    if (op.backward) {
        specs.push(getBinaryGradKernelSpec(op, op.backward, false));
        specs.push(getBinaryGradKernelSpec(op, op.backward, true));
    }
    return specs;
}

function getUnaryKernelSpecs(op: UnaryOpSpec): KernelSpec[] {
    const specs = [getUnaryKernelSpec(op), getUnaryInplaceKernelSpec(op)];
    if (op.backward) {
        specs.push(getUnaryGradKernelSpec(op, op.backward));
    }
    return specs;
}

function getReductionKernelSpec(op: ReductionOpSpec): KernelSpec {
    const initCode = exprCodeToWebGLShader(op.init, {
        input: "input[local_id.x]",
        output: "accumulator",
    });
    const forwardCode = exprCodeToWebGLShader(op.forward, {
        input: "input[i]",
        output: "accumulator",
    });
    const reduceCode =
        op.reduce === undefined
            ? ""
            : exprCodeToWebGLShader(op.reduce, {
                  input: "input[i]",
                  output: "accumulator",
                  inputSize: "parameters.size",
              });
    const shader = `
    var ${initCode};
    // Load inputData into local memory
    for (var i = local_id.x; i < parameters.size; i += $$workgroupSize$$) {
        ${forwardCode};
    }
    // Write partial group sum to outputData
    output[local_id.x] = accumulator;

    workgroupBarrier(); // Make sure all threads have completed reduction

    // First thread sums up results from all other threads
    if (local_id.x == 0u) {
        var numToSum = min(parameters.size, $$workgroupSize$$u);
        for (var i = 1u; i < numToSum; i++) {
            accumulator ${op.combineOp}= output[i];
        }
        // Store final reduction in the first element of result array
        ${reduceCode};
        output[0] = accumulator;
    }
`;
    return {
        name: op.name,
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
                size: "workgroupSize",
            },
        ],
        workgroupSize: ["workgroupSize", 1, 1],
        workgroupCount: [1, 1, 1],
        shader: shader,
    };
}

function getReductionGradKernelSpec(
    op: ReductionOpSpec,
    backwardExprCode: ExprCode
): KernelSpec {
    const backwardShaderCode = exprCodeToWebGLShader(backwardExprCode, {
        input: "input[index]",
        output: "output[0]",
        outputGrad: "outputGrad[0]",
        inputGrad: "inputGrad[index]",
        inputSize: "parameters.size",
    });
    const shader = `
    let index = global_id.x;
    if (index >= parameters.size) {
        return;
    }
    ${backwardShaderCode};
`;
    return {
        name: op.name + "_grad",
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
            {
                name: "output",
                shaderType: "array<f32>",
            },
            {
                name: "outputGrad",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "inputGrad",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: ["workgroupSize", 1, 1],
        workgroupCount: ["size/workgroupSize", 1, 1],
        shader: shader,
    };
}

function getReductionDimKernelSpec(op: ReductionOpSpec): KernelSpec {
    const initCode = exprCodeToWebGLShader(op.init, {
        input: "input[inputIndex]",
        output: "accumulator",
    });
    const forwardCode = exprCodeToWebGLShader(op.forward, {
        input: "input[inputIndex]",
        output: "accumulator",
    });
    const reduceCode =
        op.reduce === undefined
            ? ""
            : exprCodeToWebGLShader(op.reduce, {
                  input: "input[inputIndex]",
                  output: "accumulator",
                  inputSize: "dimN",
              });
    let shader = `
    let outputIndex = global_id.x;
    if (outputIndex >= parameters.size) {
        return;
    }
    var i = outputIndex;
    var outputIndex0 = u32(i / parameters.outputStride0);
    i = i % parameters.outputStride0;
    var outputIndex1 = u32(i / parameters.outputStride1);
    i = i % parameters.outputStride1;
    var outputIndex2 = u32(i / parameters.outputStride2);
    i = i % parameters.outputStride2;
    var outputIndex3 = i;
    let dimN = parameters.inputShape$$dim$$;
    var ${initCode};
    for (var dimI = 0u; dimI < dimN; dimI++) {
        outputIndex$$dim$$ = dimI;
        let inputIndex =
            outputIndex0 * parameters.inputStride0 +
            outputIndex1 * parameters.inputStride1 +
            outputIndex2 * parameters.inputStride2 +
            outputIndex3;
        ${forwardCode};
    }
    ${reduceCode};
    output[outputIndex] = accumulator;
`;
    return {
        name: op.name + "_dim",
        config: [
            {
                name: "dtype",
            },
            {
                name: "dim",
            },
            {
                name: "maxdim",
            },
        ],
        parameters: [
            {
                name: "inputShape0",
                shaderType: "u32",
            },
            {
                name: "inputShape1",
                shaderType: "u32",
            },
            {
                name: "inputShape2",
                shaderType: "u32",
            },
            {
                name: "inputShape3",
                shaderType: "u32",
            },
            {
                name: "inputStride0",
                shaderType: "u32",
            },
            {
                name: "inputStride1",
                shaderType: "u32",
            },
            {
                name: "inputStride2",
                shaderType: "u32",
            },
            {
                name: "inputStride3",
                shaderType: "u32",
            },
            {
                name: "outputStride0",
                shaderType: "u32",
            },
            {
                name: "outputStride1",
                shaderType: "u32",
            },
            {
                name: "outputStride2",
                shaderType: "u32",
            },
            {
                name: "outputStride3",
                shaderType: "u32",
            },
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
        workgroupSize: [256, 1, 1],
        workgroupCount: ["size/256", 1, 1],
        shader: shader,
    };
}

function getBinaryKernelSpec(
    op: BinaryOpSpec,
    inplace: boolean,
    isOtherScalar: boolean,
    strided: boolean
): KernelSpec {
    const maxdim = 4;
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    if (strided) {
        for (let dim = 0; dim < maxdim; dim++) {
            parameters.push({
                name: `inputStrides${dim}`,
                shaderType: "u32",
            });
            parameters.push({
                name: `otherStrides${dim}`,
                shaderType: "u32",
            });
            parameters.push({
                name: `outputStrides${dim}`,
                shaderType: "u32",
            });
        }
    }
    const subs: any = {
        input: "input[inputIndex]",
        other: "other[otherIndex]",
        output: "output[outputIndex]",
    };
    if (inplace) {
        subs.output = "input[outputIndex]";
    }
    if (isOtherScalar) {
        subs.other = "parameters.other";
        parameters.push({
            name: "other",
            shaderType: "f32",
        });
    }
    if (op.alpha !== undefined && op.alpha) {
        parameters.push({
            name: "alpha",
            shaderType: "f32",
        });
        subs["alpha"] = "parameters.alpha";
    }
    const shaderSnippet = exprCodeToWebGLShader(op.forward, subs);
    let shader: string;
    if (strided) {
        shader = `
        let outputIndex = global_id.x;
        if (outputIndex >= parameters.size) {
            return;
        }
        var i = outputIndex;
        let outputIndex0 = u32(i / parameters.outputStrides0);
        i = i % parameters.outputStrides0;
        let outputIndex1 = u32(i / parameters.outputStrides1);
        i = i % parameters.outputStrides1;
        let outputIndex2 = u32(i / parameters.outputStrides2);
        i = i % parameters.outputStrides2;
        let outputIndex3 = i;
        let inputIndex =
            outputIndex0 * parameters.inputStrides0 +
            outputIndex1 * parameters.inputStrides1 +
            outputIndex2 * parameters.inputStrides2 +
            outputIndex3;
        let otherIndex =
            outputIndex0 * parameters.otherStrides0 +
            outputIndex1 * parameters.otherStrides1 +
            outputIndex2 * parameters.otherStrides2 +
            outputIndex3;
        ${shaderSnippet};`;
    }
    else {
        shader = `
        let outputIndex = global_id.x;
        if (outputIndex >= parameters.size) {
            return;
        }
        let inputIndex = outputIndex;
        let otherIndex = outputIndex;
        ${shaderSnippet};`;
    }
    const inputs: KernelInputSpec[] = [];
    if (!isOtherScalar) {
        inputs.push({
            name: "other",
            shaderType: "array<f32>",
        });
    }
    let outputName = "input";
    if (!inplace) {
        inputs.splice(0, 0, {
            name: "input",
            shaderType: "array<f32>",
        });
        outputName = "output";
    }
    let name = op.name;
    if (isOtherScalar) {
        name += "_scalar";
    }
    if (strided) {
        name += "_strided";
    }
    if (inplace) {
        name += "_";
    }
    return {
        name: name,
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: parameters,
        inputs: inputs,
        outputs: [
            {
                name: outputName,
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["size/256", 1, 1],
        shader: shader,
    };
}

function getBinaryGradKernelSpec(
    op: BinaryOpSpec,
    backward: ExprCode,
    isOtherScalar: boolean
): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    const subs: any = {
        input: "input[global_id.x]",
        inputGrad: "inputGrad[global_id.x]",
        output: "output[global_id.x]",
        outputGrad: "outputGrad[global_id.x]",
        other: "other[global_id.x]",
        otherGrad: "otherGrad[global_id.x]",
    };
    if (isOtherScalar) {
        subs.other = "parameters.other";
        subs.otherGrad = "otherGrad";
        parameters.push({
            name: "other",
            shaderType: "f32",
        });
    }
    if (op.alpha !== undefined && op.alpha) {
        parameters.push({
            name: "alpha",
            shaderType: "f32",
        });
        subs["alpha"] = "parameters.alpha";
    }
    const ast = parseCode(backward);
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        if (global_id.x >= parameters.size) {
            return;
        }
        ${isOtherScalar? "var otherGrad = 0.0;" : ""}
        ${shaderSnippet};`;
    const inputs: KernelInputSpec[] = [
        {
            name: "input",
            shaderType: "array<f32>",
        }];
    if (!isOtherScalar) {
        inputs.push({
            name: "other",
            shaderType: "array<f32>",
        });
    }
    inputs.push({
            name: "outputGrad",
            shaderType: "array<f32>",
        });
    const outputs: KernelOutputSpec[] = [
        {
            name: "inputGrad",
            shaderType: "array<f32>",
            size: "size",
        }];
    if (!isOtherScalar) {
        outputs.push({
            name: "otherGrad",
            shaderType: "array<f32>",
            size: "size",
        });
    }
    return {
        name: op.name + (isOtherScalar ? "_scalar" : "") + "_grad",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters,
        inputs,
        outputs,
        workgroupSize: [256, 1, 1],
        workgroupCount: ["size/256", 1, 1],
        shader,
    };
}

function getUnaryKernelSpec(op: UnaryOpSpec): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
        // {
        //     name: "strideX",
        //     shaderType: "u32",
        // },
    ];
    const subs: any = {
        input: "input[index]",
        output: "output[index]",
    };
    if (op.alpha !== undefined && op.alpha) {
        parameters.push({
            name: "alpha",
            shaderType: "f32",
        });
        subs["alpha"] = "parameters.alpha";
    }
    const ast = parseCode(op.forward);
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        var index = global_id.x;
        if (index >= parameters.size) {
            return;
        }
        ${shaderSnippet};`;
    return {
        name: op.name,
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: parameters,
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
        workgroupSize: [256, 1, 1],
        workgroupCount: ["size/256", 1, 1],
        // workgroupCount: ["var sizeX = ceil(pow(size, 0.5));sizeX / 16", "var sizeX = ceil(pow(size, 0.5));ceil(size/sizeX) / 16", 1],
        // workgroupCount: [1, 1, 1],
        // workgroupVariables: [{name:"local_input", shaderType:["array<f32>", 256*8]}],
        shader: shader,
    };
}

function getUnaryInplaceKernelSpec(op: UnaryOpSpec): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    const ast = parseCode(op.forward);
    const subs: any = {
        input: "input[global_id.x]",
        output: "input[global_id.x]",
    };
    if (op.alpha !== undefined && op.alpha) {
        parameters.push({
            name: "alpha",
            shaderType: "f32",
        });
        subs["alpha"] = "parameters.alpha";
    }
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        if (global_id.x >= parameters.size) {
            return;
        }
        ${shaderSnippet};`;
    return {
        name: op.name + "_",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: parameters,
        inputs: [],
        outputs: [
            {
                name: "input",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["size/256", 1, 1],
        shader: shader,
    };
}

function getUnaryGradKernelSpec(
    op: UnaryOpSpec,
    backward: ExprCode
): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    const subs: any = {
        input: "input[global_id.x]",
        inputGrad: "inputGrad[global_id.x]",
        output: "output[global_id.x]",
        outputGrad: "outputGrad[global_id.x]",
    };
    if (op.alpha !== undefined && op.alpha) {
        parameters.push({
            name: "alpha",
            shaderType: "f32",
        });
        subs["alpha"] = "parameters.alpha";
    }
    const ast = parseCode(backward);
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        if (global_id.x >= parameters.size) {
            return;
        }
        ${shaderSnippet};`;
    return {
        name: op.name + "_grad",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: parameters,
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>",
            },
            {
                name: "outputGrad",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "inputGrad",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["size/256", 1, 1],
        shader: shader,
    };
}
