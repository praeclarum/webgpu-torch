import { KernelParamSpec, KernelSpec } from "./kernel";
import { exprNodeToString, parseCode, substituteIdentifiers } from "./expr";
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

function getBinaryKernelSpecs(op: BinaryOpSpec): KernelSpec[] {
    return [getBinaryKernelSpec(op), getBinaryInplaceKernelSpec(op)];
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
        out: "out[global_id.x]",
    };
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToString(shaderAst);
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
                name: "out",
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
        out: "input[global_id.x]",
    };
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToString(shaderAst);
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

function getUnaryKernelSpecs(op: UnaryOpSpec): KernelSpec[] {
    return [getUnaryKernelSpec(op), getUnaryInplaceKernelSpec(op)];
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
        out: "out[global_id.x]",
    };
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToString(shaderAst);
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
                name: "out",
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
        out: "input[global_id.x]",
    };
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToString(shaderAst);
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

export function getKernelSpecs(op: OpSpec): KernelSpec[] {
    if (op.type == "binary") {
        return getBinaryKernelSpecs(op as BinaryOpSpec);
    } else {
        return getUnaryKernelSpecs(op as UnaryOpSpec);
    }
}
