import { KernelParamSpec, KernelSpec } from "./kernel";
import { ExprCode, exprNodeToWebGLShader, parseCode, substituteIdentifiers } from "./expr";
import { BinaryOpSpec, OpSpec, UnaryOpSpec } from "./op_spec";

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

export function getKernelSpecs(op: OpSpec): KernelSpec[] {
    if (op.type == "binary") {
        return getBinaryKernelSpecs(op as BinaryOpSpec);
    } else {
        return getUnaryKernelSpecs(op as UnaryOpSpec);
    }
}

function getBinaryKernelSpecs(op: BinaryOpSpec): KernelSpec[] {
    const specs = [
        getBinaryKernelSpec(op),
        getBinaryInplaceKernelSpec(op),
    ];
    if (op.backward) {
        specs.push(getBinaryGradKernelSpec(op, op.backward));
    }
    return specs;
}

function getUnaryKernelSpecs(op: UnaryOpSpec): KernelSpec[] {
    const specs = [
        getUnaryKernelSpec(op),
        getUnaryInplaceKernelSpec(op),
    ];
    if (op.backward) {
        specs.push(getUnaryGradKernelSpec(op, op.backward));
    }
    return specs;
}

function getBinaryKernelSpec(op: BinaryOpSpec): KernelSpec {
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

function getBinaryGradKernelSpec(op: BinaryOpSpec, backward: ExprCode): KernelSpec {
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

function getUnaryGradKernelSpec(op: UnaryOpSpec, backward: ExprCode): KernelSpec {
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
