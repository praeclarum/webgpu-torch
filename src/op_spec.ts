import { KernelParamSpec, KernelSpec } from "./kernel";
import { exprNodeToString, parseExpr, substituteIdentifiers } from "./expr";

export type OpType = "unary" | "binary";

export type OpSpec = {
    name: string;
    aliases?: string[];
    alpha?: boolean;
    type: OpType;
    webGPU: string;
    javaScript: string;    
}

export type UnaryOpSpec = OpSpec & {
    type: "unary";
}

export type BinaryOpSpec = OpSpec & {
    type: "binary";
}

class CodeWriter {
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

function titleCase(str: string) {
    return str[0].toUpperCase() + str.slice(1);
}

export function getKernelSpecs(op: OpSpec): KernelSpec[] {
    if (op.type == "binary") {
        const parameters: KernelParamSpec[] = [
            {
                name: "size",
                shaderType: "u32",
            },
        ];
        const ast = parseExpr(op.webGPU);
        const subs = {
            input: "input[global_id.x]",
            other: "other[global_id.x]",
        };
        const shaderAst = substituteIdentifiers(ast, subs);
        const shaderSnippet = exprNodeToString(shaderAst);
        const shader = `
            if (global_id.x >= parameters.size) {
                return;
            }
            out[global_id.x] = ${shaderSnippet};`;
        const kernels: KernelSpec[] = [{
            name: titleCase(op.name),
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
        }];
        return kernels;
    }
    else {
        throw new Error("Unary Op Specs Not implemented");
    }
}

