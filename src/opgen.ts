// Generate code from op_spec.ts and op_table.ts
import { KernelParamSpec, KernelSpec } from "./kernel";
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
    const specs = [getReductionKernelSpec(op)];
    if (op.backward) {
        // specs.push(getReductionGradKernelSpec(op, op.backward));
    }
    return specs;
}

function getBinaryKernelSpecs(op: BinaryOpSpec): KernelSpec[] {
    const specs = [getBinaryKernelSpec(op), getBinaryInplaceKernelSpec(op)];
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
        input: "input[global_id.x]",
        output: "accumulator",
    });
    const forwardCode = exprCodeToWebGLShader(op.forward, {
        input: "input[i]",
        output: "accumulator",
    });
    const reduceCode = op.reduce === undefined ? "" : exprCodeToWebGLShader(op.reduce, {
        input: "input[i]",
        output: "accumulator",
    });
    const shader = `
    var ${initCode};
    // Load inputData into local memory
    for (var i = local_id.x; i < parameters.size; i += $$workgroupSize$$) {
        ${forwardCode};
    }
    // Write partial group sum to outputData
    output[local_id.x] = accumulator;

    workgroupBarrier(); // Make sure all threads have completed summation

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
        workgroupCount: ["size/workgroupSize", 1, 1],
        shader: shader,
    };
}

function getBinaryKernelSpec(op: BinaryOpSpec): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    const subs = {
        input: "input[global_id.x]",
        other: "other[global_id.x]",
        output: "output[global_id.x]",
    };
    const shaderSnippet = exprCodeToWebGLShader(op.forward, subs);
    const shader = `
        if (global_id.x >= parameters.size) {
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
            {
                name: "other",
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
        workgroupSize: [64, 1, 1],
        workgroupCount: ["size/8", 1, 1],
        shader: shader,
    };
}

function getBinaryInplaceKernelSpec(op: BinaryOpSpec): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    const ast = parseCode(op.forward);
    const subs = {
        input: "input[global_id.x]",
        other: "other[global_id.x]",
        output: "input[global_id.x]",
    };
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
        inputs: [
            {
                name: "other",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "input",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: [64, 1, 1],
        workgroupCount: ["size/8", 1, 1],
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
    const ast = parseCode(backward);
    const subs = {
        input: "input[global_id.x]",
        inputGrad: "inputGrad[global_id.x]",
        output: "output[global_id.x]",
        outputGrad: "outputGrad[global_id.x]",
        other: "other[global_id.x]",
        otherGrad: "otherGrad[global_id.x]",
    };
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        if (global_id.x >= parameters.size) {
            return;
        }
        ${shaderSnippet};`;
    return {
        name: op.name + "Grad",
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
        workgroupSize: [64, 1, 1],
        workgroupCount: ["size/8", 1, 1],
        shader: shader,
    };
}

function getUnaryKernelSpec(op: UnaryOpSpec): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    const ast = parseCode(op.forward);
    const subs = {
        input: "input[global_id.x]",
        output: "output[global_id.x]",
    };
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        if (global_id.x >= parameters.size) {
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
        workgroupSize: [64, 1, 1],
        workgroupCount: ["size/8", 1, 1],
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
    const subs = {
        input: "input[global_id.x]",
        output: "input[global_id.x]",
    };
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
        workgroupSize: [64, 1, 1],
        workgroupCount: ["size/8", 1, 1],
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
    const ast = parseCode(backward);
    const subs = {
        input: "input[global_id.x]",
        inputGrad: "inputGrad[global_id.x]",
        output: "output[global_id.x]",
        outputGrad: "outputGrad[global_id.x]",
    };
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        if (global_id.x >= parameters.size) {
            return;
        }
        ${shaderSnippet};`;
    return {
        name: op.name + "Grad",
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
        workgroupSize: [64, 1, 1],
        workgroupCount: ["size/8", 1, 1],
        shader: shader,
    };
}
