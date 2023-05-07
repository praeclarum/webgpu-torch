import { KernelSpec } from "../src/kernel";
import { CodeWriter, getKernelSpecs } from "../src/opgen";
import { OpSpec } from "../src/op_spec";
import { registry } from "../src/op_table";

// import fs
import * as fs from "fs";

console.log("Running code generator...");

const absSrcDir = fs.realpathSync(__dirname + "/../src");
console.log("src dir:", absSrcDir);

// Generate op kernels
const kernelsSpecsIndex: {[name: string]: KernelSpec} = {};
const kernelsSpecs: [OpSpec, KernelSpec][] = [];
for (const spec of registry) {
    const kernels = getKernelSpecs(spec);
    for (const kernel of kernels) {
        kernelsSpecsIndex[kernel.name] = kernel;
        kernelsSpecs.push([spec, kernel]);
    }
}

// Write the kernels
function writeOpKernelsCode(): void {
    const outPath = absSrcDir + "/kernels_opgen.ts";
    const json = JSON.stringify(kernelsSpecsIndex, null, 4);
    console.log("Writing", outPath);
    fs.writeFileSync(outPath, `// Generated by scripts/codegen.ts
// Do not edit this file directly.
import { KernelSpec } from "./kernel";

export const kernels: { [name: string]: KernelSpec } =
${json};
`);
}
writeOpKernelsCode();

function insertCodegenIntoFile(path: string, codegen: string): void {
    const code = fs.readFileSync(path, "utf8");
    const codegenMarker = "// Codegen marker";
    const endCodegenMarker = "// End codegen marker";
    const startIndex = code.indexOf(codegenMarker);
    const endIndex = code.indexOf(endCodegenMarker);
    if (startIndex === -1 || endIndex === -1) {
        throw new Error("Could not find codegen marker in " + path);
    }
    const pre = code.slice(0, startIndex + codegenMarker.length);
    const post = code.slice(endIndex);
    const newCode = pre + "\n" + codegen + "\n    " + post;
    console.log("Writing", path);
    fs.writeFileSync(path, newCode);
}

// Write the TensorImpl class
function writeTensorImplCode(): void {
    const w = new CodeWriter();
    w.indent();
    for (const [opSpec, kernelSpec] of kernelsSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        const isGrad = kernelSpec.name.endsWith("Grad");
        if (isGrad) continue;
        const isBinary = opSpec.type === "binary";
        const hasAlpha = opSpec.alpha ?? false;
        if (isBinary) {
            if (hasAlpha) {
                w.writeLine(`${kernelSpec.name}(other: TensorImpl, alpha?: number): TensorImpl {`);
            }
            else {
                w.writeLine(`${kernelSpec.name}(other: TensorImpl): TensorImpl {`);
            }
            w.indent();
            w.writeLine(`const params = {`);
            w.indent();
            w.writeLine(`size: shapeSize(this.shape),`);
            if (hasAlpha) {
                w.writeLine(`alpha: alpha || 1.0,`);
            }
            w.dedent();
            w.writeLine(`};`);
            if (isInplace) {
                w.writeLine(`return this.runKernelInplace("${kernelSpec.name}", { dtype: this.dtype }, params, other);`);
            }
            else {
                w.writeLine(`return this.runKernel("${kernelSpec.name}", { dtype: this.dtype }, params, [this.shape], other)[0];`);
            }
            w.dedent();
            w.writeLine(`}`);
        }
        else {
            if (hasAlpha) {
                w.writeLine(`${kernelSpec.name}(alpha?: number): TensorImpl {`);
            }
            else {
                w.writeLine(`${kernelSpec.name}(): TensorImpl {`);
            }
            w.indent();
            w.writeLine(`const params = {`);
            w.indent();
            w.writeLine(`size: shapeSize(this.shape),`);
            if (hasAlpha) {
                w.writeLine(`alpha: alpha || 1.0,`);
            }
            w.dedent();
            w.writeLine(`};`);
            if (isInplace) {
                w.writeLine(`return this.runKernelInplace("${kernelSpec.name}", { dtype: this.dtype }, params);`);
            }
            else {
                w.writeLine(`return this.runKernel("${kernelSpec.name}", { dtype: this.dtype }, params, [this.shape])[0];`);
            }
            w.dedent();
            w.writeLine(`}`);
        }
    }
    const code = w.toString();
    // console.log(code);
    const path2 = absSrcDir + "/tensor_impl.ts";
    insertCodegenIntoFile(path2, "");
}
writeTensorImplCode();

// Write the Tensor class
function writeTensorCode(): void {
    const w = new CodeWriter();
    w.indent();
    for (const [opSpec, kernelSpec] of kernelsSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        const isGrad = kernelSpec.name.endsWith("Grad");
        if (isGrad) continue;
        const isBinary = opSpec.type === "binary";
        const hasAlpha = opSpec.alpha ?? false;
        const writeHeader = (name: string) => {
            if (isBinary) {
                if (hasAlpha) {
                    w.writeLine(`${name}(other: Tensor, alpha?: number): Tensor {`);
                }
                else {
                    w.writeLine(`${name}(other: Tensor): Tensor {`);
                }
            }
            else {
                if (hasAlpha) {
                    w.writeLine(`${name}(alpha?: number): Tensor {`);
                }
                else {
                    w.writeLine(`${name}(): Tensor {`);
                }
            }
        };
        writeHeader(kernelSpec.name);
        w.indent();
        if (isBinary) {
            if (hasAlpha) {
                if (isInplace) {
                    w.writeLine(`this._impl.${kernelSpec.name}(other._impl, alpha);`);
                    w.writeLine(`return this;`);
                }
                else {
                    w.writeLine(`return ops.${kernelSpec.name}(this, other, alpha);`);
                }
                w.dedent();
                w.writeLine(`}`);
            }
            else {
                if (isInplace) {
                    w.writeLine(`this._impl.${kernelSpec.name}(other._impl);`);
                    w.writeLine(`return this;`);
                }
                else {
                    w.writeLine(`return ops.${kernelSpec.name}(this, other);`);
                }
                w.dedent();
                w.writeLine(`}`);
            }
        }
        else {
            if (hasAlpha) {
                if (isInplace) {
                    w.writeLine(`this._impl.${kernelSpec.name}(alpha);`);
                    w.writeLine(`return this;`);
                }
                else {
                    w.writeLine(`return ops.${kernelSpec.name}(this, alpha);`);
                }
                w.dedent();
                w.writeLine(`}`);
            }
            else {
                if (isInplace) {
                    w.writeLine(`this._impl.${kernelSpec.name}();`);
                    w.writeLine(`return this;`);
                }
                else {
                    w.writeLine(`return ops.${kernelSpec.name}(this);`);
                }
                w.dedent();
                w.writeLine(`}`);
            }
        }
        if (!isInplace) {
            for (const alias of opSpec.aliases ?? []) {
                writeHeader(alias);
                w.indent();
                if (isBinary) {
                    if (hasAlpha) {
                        w.writeLine(`return ops.${kernelSpec.name}(this, other, alpha);`);
                    }
                    else {
                        w.writeLine(`return ops.${kernelSpec.name}(this, other);`);
                    }
                }
                else {
                    if (hasAlpha) {
                        w.writeLine(`return ops.${kernelSpec.name}(this, alpha);`);
                    }
                    else {
                        w.writeLine(`return ops.${kernelSpec.name}(this);`);
                    }
                }
                w.dedent();
                w.writeLine(`}`);
            }
        }
    }
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/tensor.ts";
    insertCodegenIntoFile(path, code);
}
writeTensorCode();

// Write the ITensor interface
function writeITensorCode(): void {
    const w = new CodeWriter();
    w.indent();
    for (const [opSpec, kernelSpec] of kernelsSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        const isGrad = kernelSpec.name.endsWith("Grad");
        if (isGrad) continue;
        const isBinary = opSpec.type === "binary";
        const hasAlpha = opSpec.alpha ?? false;
        if (isBinary) {
            if (hasAlpha) {
                w.writeLine(`${kernelSpec.name}(other: ITensor, alpha?: number): ITensor;`);
            }
            else {
                w.writeLine(`${kernelSpec.name}(other: ITensor): ITensor;`);
            }
        }
        else {
            if (hasAlpha) {
                w.writeLine(`${kernelSpec.name}(alpha?: number): ITensor;`);
            }
            else {
                w.writeLine(`${kernelSpec.name}(): ITensor;`);
            }
        }
    }
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/tensor_if.ts";
    insertCodegenIntoFile(path, code);
}
writeITensorCode();

// Write autograd functions
function writeFunctionsCode(): void {
    const w = new CodeWriter();
    w.writeLine(`import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import { Tensor } from "./tensor";
import { shapeSize } from "./shape";`);
    for (const [opSpec, kernelSpec] of kernelsSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        if (isInplace) {
            continue;
        }
        const isGrad = kernelSpec.name.endsWith("Grad");
        if (isGrad) continue;
        const isBinary = opSpec.type === "binary";
        const hasAlpha = opSpec.alpha ?? false;
        const className = kernelSpec.name[0].toUpperCase() + kernelSpec.name.slice(1) + "Function";
        const writeUnpackInputs = (inputsName: string, includeAlpha: boolean) => {
            if (isBinary) {
                if (hasAlpha && includeAlpha) {
                    w.writeLine(`const [input, other, alpha] = ${inputsName} as [Tensor, Tensor, number|undefined];`);
                }
                else {
                    w.writeLine(`const [input, other] = ${inputsName} as [Tensor, Tensor];`);
                }
            }
            else {
                if (hasAlpha && includeAlpha) {
                    w.writeLine(`const [input, alpha] = ${inputsName} as [Tensor, number|undefined];`);
                }
                else {
                    w.writeLine(`const [input] = ${inputsName} as [Tensor];`);
                }
            }
        }
        const writeParams = (alphaName: string) => {
            w.writeLine(`const params = {`);
            w.indent();
            w.writeLine(`size: shapeSize(input.shape),`);
            if (hasAlpha) {
                w.writeLine(`alpha: ${alphaName} || 1.0,`);
            }
            w.dedent();
            w.writeLine(`};`);
        };
        w.writeLine(`export class ${className} extends AutoFunction {`);
        w.indent();

        // Forward
        w.writeLine(`static forward(inputs: FunctionInput[]): Tensor {`);
        w.indent();
        writeUnpackInputs("inputs", true);
        writeParams("alpha");
        w.writeLine(`if (!input.isContiguous) { throw new Error("Input must be contiguous"); }`);
        if (isBinary) {
            w.writeLine(`if (!other.isContiguous) { throw new Error("Other must be contiguous"); }`);
            w.writeLine(`return input.runKernel("${kernelSpec.name}", { dtype: input.dtype }, params, [input.shape], other)[0];`);
        }
        else {
            w.writeLine(`return input.runKernel("${kernelSpec.name}", { dtype: input.dtype }, params, [input.shape])[0];`);
        }
        w.dedent();
        w.writeLine(`}`);

        w.writeLine(`static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {`);
        w.indent();
        writeUnpackInputs("inputs", true);
        if (isBinary) {
            if (hasAlpha) {
                w.writeLine(`ctx.alpha = alpha;`);
            }
            w.writeLine(`ctx.saveForBackward(input, other);`);
        }
        else {
            if (hasAlpha) {
                w.writeLine(`ctx.alpha = alpha;`);
            }
            w.writeLine(`ctx.saveForBackward(input);`);
        }
        w.dedent();
        w.writeLine(`}`);

        // Backward
        w.writeLine(`static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {`);
        w.indent();
        writeUnpackInputs("ctx.savedTensors", false);
        writeParams("ctx.alpha");
        if (isBinary) {
            w.writeLine(`return input.runKernel("${kernelSpec.name}Grad", { dtype: input.dtype }, params, [input.shape, other.shape], other, outputGrad);`);
        }
        else {
            w.writeLine(`return input.runKernel("${kernelSpec.name}Grad", { dtype: input.dtype }, params, [input.shape], outputGrad);`);
        }
        w.dedent();
        w.writeLine(`}`);
        w.dedent();
        w.writeLine(`}`);
    }
    w.writeLine("");
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/functions_opgen.ts";
    console.log("Writing", path);
    fs.writeFileSync(path, code);
}
writeFunctionsCode();

// Write global ops
function writeOpsCode(): void {
    const w = new CodeWriter();
    w.writeLine(`import * as functions from "./functions";
import { Tensor } from "./tensor";
import { unary, unaryWithAlpha, binary, binaryWithAlpha } from "./ops_high";`);
    for (const [opSpec, kernelSpec] of kernelsSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        if (isInplace) {
            continue;
        }
        const isGrad = kernelSpec.name.endsWith("Grad");
        if (isGrad) continue;
        const isBinary = opSpec.type === "binary";
        const hasAlpha = opSpec.alpha ?? false;
        const funcName = kernelSpec.name[0].toUpperCase() + kernelSpec.name.slice(1) + "Function";
        const writeHeader = (name: string) => {
            if (isBinary) {
                if (hasAlpha) {
                    w.writeLine(`export function ${name}(input: Tensor, other: Tensor, alpha?: number): Tensor {`);
                }
                else {
                    w.writeLine(`export function ${name}(input: Tensor, other: Tensor): Tensor {`);
                }
            }
            else {
                if (hasAlpha) {
                    w.writeLine(`export function ${name}(input: Tensor, alpha?: number): Tensor {`);
                }
                else {
                    w.writeLine(`export function ${name}(input: Tensor): Tensor {`);
                }
            }
        };
        writeHeader(kernelSpec.name);
        w.indent();
        if (isBinary) {
            if (hasAlpha) {
                w.writeLine(`return binaryWithAlpha(functions.${funcName}, input, other, alpha);`);
            }
            else {
                w.writeLine(`return binary(functions.${funcName}, input, other);`);
            }
        }
        else {
            if (hasAlpha) {
                w.writeLine(`return unaryWithAlpha(functions.${funcName}, input, alpha);`);
            }
            else {
                w.writeLine(`return unary(functions.${funcName}, input);`);
            }
        }
        w.dedent();
        w.writeLine(`}`);
        for (const alias of opSpec.aliases ?? []) {
            writeHeader(alias);
            w.indent();
            if (isBinary) {
                if (hasAlpha) {
                    w.writeLine(`return ${kernelSpec.name}(input, other, alpha);`);
                }
                else {
                    w.writeLine(`return ${kernelSpec.name}(input, other);`);
                }
            }
            else {
                if (hasAlpha) {
                    w.writeLine(`return ${kernelSpec.name}(input, alpha);`);
                }
                else {
                    w.writeLine(`return ${kernelSpec.name}(input);`);
                }
            }
            w.dedent();
            w.writeLine(`}`);
        }
    }
    w.writeLine("");
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/ops_opgen.ts";
    console.log("Writing", path);
    fs.writeFileSync(path, code);
}
writeOpsCode();
