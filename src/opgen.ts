// Generate code from op_spec.ts and op_table.ts
import { KernelInputSpec, KernelParamSpec, KernelSpec } from "./kernel";
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
        getBinaryKernelSpec(op, false, false),
        getBinaryKernelSpec(op, true, false),
        getBinaryKernelSpec(op, false, true),
        getBinaryKernelSpec(op, true, true),
    ];
    if (op.backward) {
        specs.push(getBinaryGradKernelSpec(op, op.backward));
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
    var index = global_id.x;
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
        other: "other[global_id.x]",
        output: "output[global_id.x]",
    };
    if (inplace) {
        subs.output = "input[global_id.x]";
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
    const shader = `
        if (global_id.x >= parameters.size) {
            return;
        }
        ${shaderSnippet};`;
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
        other: "other[global_id.x]",
        otherGrad: "otherGrad[global_id.x]",
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
                name: "other",
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
            {
                name: "otherGrad",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["size/256", 1, 1],
        shader: shader,
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
