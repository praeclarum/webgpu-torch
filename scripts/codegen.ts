import { CodeWriter } from "../src/opgen";
import { OpSpec, ReductionOpSpec } from "../src/op_spec";
import { opKernelSpecs } from "../src/kernels_opgen";
import { registry as opRegistry } from "../src/op_table";

// import fs
import * as fs from "fs";

console.log("Running code generator...");

const absSrcDir = fs.realpathSync(__dirname + "/../src");
console.log("src dir:", absSrcDir);

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
    writeFile(path, newCode);
}

function writeOpHeader(opSpec: OpSpec, name: string, isAlias: boolean, suffix: string, w: CodeWriter) {
    const hasAlpha = opSpec.alpha ?? false;
    const isBinary = opSpec.type === "binary";
    const isReduction = opSpec.type === "reduction";
    writeOpDocs(opSpec, "this", isAlias, w);
    if (isReduction) {
        w.writeLine(`${name}(dim?: number | number[], keepdim?: boolean): Tensor${suffix}`);
    }
    else if (isBinary) {
        if (hasAlpha) {
            w.writeLine(`${name}(other: number | Tensor, alpha?: number): Tensor${suffix}`);
        }
        else {
            w.writeLine(`${name}(other: number | Tensor): Tensor${suffix}`);
        }
    }
    else {
        if (hasAlpha) {
            w.writeLine(`${name}(alpha?: number): Tensor${suffix}`);
        }
        else {
            w.writeLine(`${name}(): Tensor${suffix}`);
        }
    }
};

function writeParams(inputName: string, otherIsScalar: boolean, hasAlpha: boolean, alphaName: string, w: CodeWriter) {
    w.writeLine(`const params = {`);
    w.indent();
    w.writeLine(`size: shapeSize(${inputName}.shape),`);
    // w.writeLine(`strideX: 1,`);
    if (otherIsScalar) {
        w.writeLine(`other: other,`);
    }
    if (hasAlpha) {
        w.writeLine(`alpha: ${alphaName} || 1.0,`);
    }
    w.dedent();
    w.writeLine(`};`);
}

function writeReductionParams(sizeShapeName: string, dimName: string | null, w: CodeWriter) {
    w.writeLine(`const params = {`);
    w.indent();
    w.writeLine(`size: shapeSize(${sizeShapeName}),`);
    if (dimName) {
        for (let dim = 0; dim < 4; dim++) {
            w.writeLine(`inputShape${dim}: input.shape.length > ${dim} ? input.shape[${dim}] : 1,`);
            w.writeLine(`inputStride${dim}: input.shape.length > ${dim} ? input.strides[${dim}] : 1,`);
            w.writeLine(`outputStride${dim}: outputShape.length > ${dim} ? outputStrides[${dim}] : 1,`);
        }
    }
    w.dedent();
    w.writeLine(`};`);
}

// Write the Tensor class
function writeTensorCode(): void {
    const w = new CodeWriter();
    w.indent();
    for (const [opSpec, kernelSpec] of opKernelSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        const isGrad = kernelSpec.name.endsWith("_grad");
        if (isGrad) continue;
        const isStrided = kernelSpec.name.includes("_strided");
        if (isStrided) continue;
        const isOtherScalar = kernelSpec.name.includes("_scalar");
        if (isOtherScalar) continue;
        const isBinary = opSpec.type === "binary";
        const hasAlpha = opSpec.alpha ?? false;
        const isReduction = opSpec.type === "reduction";
        if (isReduction && kernelSpec.name.endsWith("_dim")) continue;
        writeOpHeader(opSpec, kernelSpec.name, false, " {", w);
        w.indent();
        if (isInplace) {
            if (isBinary) {
                w.writeLine(`if (typeof other === "number") {`);
                w.indent();
                writeParams("this", true, hasAlpha, "alpha", w);
                w.writeLine(`return this.runKernelInplace("${kernelSpec.name.replace('_', '_scalar_')}", { dtype: this.dtype }, params);`);
                w.dedent();
                w.writeLine(`} else {`);
                w.indent();
                w.writeLine(`const broadcasted = broadcastShapes(this, other);`);
                w.writeLine(`if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {`);
                w.indent();
                // Build broadcasted params
                const maxdim = 4;
                w.writeLine(`const inputDims = broadcasted.a.shape.length;`);
                w.writeLine(`const otherDims = broadcasted.b.shape.length;`);
                w.writeLine(`if (inputDims > ${maxdim} || otherDims > ${maxdim}) {`);
                w.indent();
                w.writeLine(`throw new Error("Broadcasting not supported for tensors with more than ${maxdim} dimensions");`);
                w.dedent();
                w.writeLine(`}`);
                w.writeLine(`const params = {`);
                w.indent();
                for (let dim = 0; dim < maxdim; ++dim) {
                    w.writeLine(`inputStrides${dim}: inputDims > ${dim} ? broadcasted.a.strides[${dim}] : 0,`);
                    w.writeLine(`otherStrides${dim}: otherDims > ${dim} ? broadcasted.b.strides[${dim}] : 0,`);
                    w.writeLine(`outputStrides${dim}: broadcasted.output.shape.length > ${dim} ? broadcasted.output.strides[${dim}] : 1,`);
                }
                w.writeLine(`size: shapeSize(broadcasted.output.shape),`);
                if (hasAlpha) {
                    w.writeLine(`alpha: alpha || 1.0,`);
                }
                w.dedent();
                w.writeLine(`};`);
                const nameWithoutTrailingUnderscore = kernelSpec.name.slice(0, -1);
                w.writeLine(`return this.runKernelInplace("${nameWithoutTrailingUnderscore}_strided_", { dtype: this.dtype }, params, other);`);
                w.dedent();
                w.writeLine(`} else {`);
                w.indent();
                // Build contiguous params
                writeParams("this", false, hasAlpha, "alpha", w);
                w.writeLine(`return this.runKernelInplace("${kernelSpec.name}", { dtype: this.dtype }, params, other);`);
                w.dedent();
                w.writeLine(`}`);
                w.dedent();
                w.writeLine(`}`);
            }
            else {
                writeParams("this", isOtherScalar, hasAlpha, "alpha", w);
                w.writeLine(`return this.runKernelInplace("${kernelSpec.name}", { dtype: this.dtype }, params);`);
            }
        }
        else {
            if (isReduction) {
                w.writeLine(`return ops.${kernelSpec.name}(this, dim, keepdim);`);
            }
            else if (isBinary) {
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
        }
        w.dedent();
        w.writeLine(`}`);
        if (!isInplace) {
            for (const alias of opSpec.aliases ?? []) {
                writeOpHeader(opSpec, alias, true, " {", w);
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

// Write the Tensor class
function writeTensorDeclCode(): void {
    const w = new CodeWriter();
    w.indent();
    for (const [opSpec, kernelSpec] of opKernelSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        const isGrad = kernelSpec.name.endsWith("_grad");
        if (isGrad) continue;
        const isStrided = kernelSpec.name.includes("_strided");
        if (isStrided) continue;
        const isOtherScalar = kernelSpec.name.includes("_scalar");
        if (isOtherScalar) continue;
        writeOpHeader(opSpec, kernelSpec.name, false, ";", w);
        if (!isInplace) {
            for (const alias of opSpec.aliases ?? []) {
                writeOpHeader(opSpec, alias, true, ";", w);
            }
        }
    }
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/tensor.d.ts";
    // insertCodegenIntoFile(path, code);
}
writeTensorDeclCode();

// Write autograd functions
function writeFunctionsCode(): void {
    const w = new CodeWriter();
    w.writeLine(`import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import type { Tensor } from "./tensor";
import { shapeSize, defaultStrides, broadcastShapes, stridedShapeIsContiguous } from "./shape";`);
    for (const [opSpec, kernelSpec] of opKernelSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        if (isInplace) {
            continue;
        }
        const isGrad = kernelSpec.name.endsWith("_grad");
        if (isGrad) continue;
        const isStrided = kernelSpec.name.includes("_strided");
        if (isStrided) continue;
        const isScalarKernel = kernelSpec.name.includes("_scalar");
        if (isScalarKernel) continue;
        const isBinary = opSpec.type === "binary";
        const isReduction = opSpec.type === "reduction";
        const hasAlpha = opSpec.alpha ?? false;
        const className = kernelSpec.name[0].toUpperCase() + kernelSpec.name.slice(1) + "Function";
        const config: {[name: string]: number|string} = {dtype: "float32"};
        let outputShapesS: string = "[input.shape]";
        if (isReduction) {
            outputShapesS = "[[]]";
            if (kernelSpec.config.find(x => x.name==="dim")) {
                continue;
            }
            else {
                config["workgroupSize"] = 256;
            }
        }
        const configS = JSON.stringify(config);
        const writeUnpackInputs = (inputsName: string, includeAlpha: boolean) => {
            if (isReduction) {
                w.writeLine(`let [input, dim, keepdim] = ${inputsName} as [Tensor, number | number[] | undefined, boolean | undefined];`);
            }
            else if (isBinary) {
                if (hasAlpha && includeAlpha) {
                    w.writeLine(`const [input, other, alpha] = ${inputsName} as [Tensor, Tensor, number | undefined];`);
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
        const writeUnpackContext = (inputsName: string, includeAlpha: boolean) => {
            if (isReduction) {
                w.writeLine(`const [input, output] = ${inputsName}.savedTensors as [Tensor, Tensor];`);
                w.writeLine(`const dim: number | number[] | undefined = ${inputsName}.dim;`);
                w.writeLine(`const keepdim: boolean | undefined = ${inputsName}.keepdim;`);
            }
            else if (isBinary) {
                w.writeLine(`const [input, other] = ${inputsName}.savedTensors as [Tensor, Tensor];`);
                if (hasAlpha && includeAlpha) {
                    w.writeLine(`const alpha: number | undefined = ${inputsName}.alpha;`);
                }
            }
            else {
                w.writeLine(`const [input] = ${inputsName}.savedTensors as [Tensor];`);
                if (hasAlpha && includeAlpha) {
                    w.writeLine(`const alpha: number | undefined = ${inputsName}.alpha;`);
                }
            }
        }
        w.writeLine(`export class ${className} extends AutoFunction {`);
        w.indent();

        // Forward
        w.writeLine(`static forward(inputs: FunctionInput[]): Tensor {`);
        w.indent();
        writeUnpackInputs("inputs", true);
        
        if (isReduction) {
            w.writeLine(`if (dim !== undefined) {`);
            w.indent();
            w.writeLine(`dim = Array.isArray(dim) && dim.length === 1 ? dim[0] : dim;`);
            w.writeLine(`if (typeof dim === "number") {`);
            w.indent();
            w.writeLine(`const inputShape = input.shape;`);
            w.writeLine(`let outputShape = input.shape.slice();`);
            w.writeLine(`outputShape[dim] = 1;`);
            w.writeLine(`let outputStrides = defaultStrides(outputShape);`);
            writeReductionParams("outputShape", "dim", w);
            w.writeLine(`if (!keepdim) outputShape.splice(dim, 1);`);
            w.writeLine(`return input.runKernel(\"${kernelSpec.name}_dim\", {dim,maxdim:inputShape.length,dtype:\"${config.dtype}\"}, params, [outputShape])[0];`);
            w.dedent();
            w.writeLine(`} else {`);
            w.indent();
            w.writeLine(`throw new Error("Multi-dimension reduction not supported");`);
            w.dedent();
            w.writeLine(`}`);
            w.dedent();
            w.writeLine(`} else {`);
            w.indent();
            writeReductionParams("input.shape", null, w);
            w.writeLine(`return input.runKernel(\"${kernelSpec.name}\", ${configS}, params, ${outputShapesS})[0];`);
            w.dedent();
            w.writeLine(`}`);
        }
        else if (isBinary) {
            w.writeLine(`if (typeof other === "number") {`);
            w.indent();
            writeParams("input", true, hasAlpha, "alpha", w);
            w.writeLine(`return input.runKernel("${kernelSpec.name}_scalar", ${configS}, params, ${outputShapesS})[0];`);
            w.dedent();
            w.writeLine(`} else {`);
            w.indent();
            w.writeLine(`const broadcasted = broadcastShapes(input, other);`);
            w.writeLine(`if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {`);
            w.indent();
            // Build broadcasted params
            const maxdim = 4;
            w.writeLine(`const inputDims = broadcasted.a.shape.length;`);
            w.writeLine(`const otherDims = broadcasted.b.shape.length;`);
            w.writeLine(`if (inputDims > ${maxdim} || otherDims > ${maxdim}) {`);
            w.indent();
            w.writeLine(`throw new Error("Broadcasting not supported for tensors with more than ${maxdim} dimensions");`);
            w.dedent();
            w.writeLine(`}`);
            w.writeLine(`const params = {`);
            w.indent();
            for (let dim = 0; dim < maxdim; ++dim) {
                w.writeLine(`inputStrides${dim}: inputDims > ${dim} ? broadcasted.a.strides[${dim}] : 0,`);
                w.writeLine(`otherStrides${dim}: otherDims > ${dim} ? broadcasted.b.strides[${dim}] : 0,`);
                w.writeLine(`outputStrides${dim}: broadcasted.output.shape.length > ${dim} ? broadcasted.output.strides[${dim}] : 1,`);
            }
            w.writeLine(`size: shapeSize(broadcasted.output.shape),`);
            if (hasAlpha) {
                w.writeLine(`alpha: alpha || 1.0,`);
            }
            w.dedent();
            w.writeLine(`};`);
            w.writeLine(`return input.runKernel("${kernelSpec.name}_strided", ${configS}, params, [broadcasted.output.shape], other)[0];`);
            w.dedent();
            w.writeLine(`} else {`);
            w.indent();
            // Build contiguous params
            w.writeLine(`if (shapeSize(input.shape) !== shapeSize(other.shape)) {`);
            w.indent();
            w.writeLine(`throw new Error(\`Shape sizes must match. Got \${input.shape} and \${other.shape}\`);`);
            w.dedent();
            w.writeLine(`}`);
            writeParams("input", false, hasAlpha, "alpha", w);
            w.writeLine(`return input.runKernel("${kernelSpec.name}", ${configS}, params, ${outputShapesS}, other)[0];`);
            w.dedent();
            w.writeLine(`}`);
            w.dedent();
            w.writeLine(`}`);
        }
        else {
            writeParams("input", false, hasAlpha, "alpha", w);
            w.writeLine(`return input.runKernel("${kernelSpec.name}", ${configS}, params, ${outputShapesS})[0];`);
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
        if (isReduction) {
            w.writeLine(`ctx.dim = dim;`);
            w.writeLine(`ctx.keepdim = keepdim;`);
            w.writeLine(`ctx.saveForBackward(input, output);`);
        }
        else if (isBinary) {
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
        writeUnpackContext("ctx", false);
        if (isReduction) {
            w.writeLine(`if (dim !== undefined) {`);
            w.indent();
            w.writeLine(`if (typeof dim === "number") {`);
            w.indent();
            writeReductionParams("input.shape", null, w);
            w.writeLine(`return input.runKernel("${kernelSpec.name}_dim_grad", ${configS}, params, [input.shape], output, outputGrad);`);
            w.dedent();
            w.writeLine(`} else {`);
            w.indent();
            w.writeLine(`throw new Error("Multi-dimension backward reduction not supported");`);
            w.dedent();
            w.writeLine(`}`);
            w.dedent();
            w.writeLine(`} else {`);
            w.indent();
            writeReductionParams("input.shape", null, w);
            w.writeLine(`return input.runKernel("${kernelSpec.name}_grad", ${configS}, params, [input.shape], output, outputGrad);`);
            w.dedent();
            w.writeLine(`}`);
        }
        else if (isBinary) {
            w.writeLine(`if (typeof other === "number") {`);
            w.indent();
            writeParams("input", true, hasAlpha, "ctx.alpha", w);
            w.writeLine(`return input.runKernel("${kernelSpec.name}_scalar_grad", ${configS}, params, [input.shape], outputGrad);`);
            w.dedent();
            w.writeLine(`} else {`);
            w.indent();
            writeParams("input", false, hasAlpha, "ctx.alpha", w);
            w.writeLine(`return input.runKernel("${kernelSpec.name}_grad", ${configS}, params, [input.shape, other.shape], other, outputGrad);`);
            w.dedent();
            w.writeLine(`}`);
        }
        else {
            writeParams("input", false, hasAlpha, "ctx.alpha", w);
            w.writeLine(`return input.runKernel("${kernelSpec.name}_grad", ${configS}, params, [input.shape], outputGrad);`);
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
    writeFile(path, code);
}
function writeFile(path: string, code: string) {
    const oldCode = fs.existsSync(path) ? fs.readFileSync(path, { encoding: "utf8" }) : null;
    if (oldCode === code) {
        console.log("OK", path);
    }
    else {
        console.log("Writing", path);
        fs.writeFileSync(path, code, { encoding: "utf8" });
    }
}
writeFunctionsCode();

function writeOpDocs(opSpec: OpSpec, inputName: string, isAlias: boolean, w: CodeWriter, includeParamsAndReturn: boolean = true): void {
    const isBinary = opSpec.type === "binary";
    const isUnary = opSpec.type === "unary";
    const hasAlpha = opSpec.alpha ?? false;
    w.writeLine(`/**`);
    if (isAlias) {
        w.writeLine(`* Alias for \`${opSpec.name}\`.`);
        w.writeLine(`*`);
    }
    if (isUnary) {
        w.writeLine(`* ![Plot of ${opSpec.name} and its gradient](../../plots/${opSpec.name}.svg)`);
        w.writeLine(`*`);
    }
    w.writeLine(`* Calculates:`);
    w.writeLine(`* \`\`\`js`);
    w.writeLine(`* ${opSpec.forward}`);
    w.writeLine(`* \`\`\``);
    w.writeLine(`*`);
    if (opSpec.type === "reduction") {
        w.writeLine(`* with an initial value of \`${(opSpec as ReductionOpSpec).init}\`.`);
        w.writeLine(`*`);
    }
    if (opSpec.backward) {
        w.writeLine(`* Gradient:`);
        w.writeLine(`* \`\`\`js`);
        w.writeLine(`* ${opSpec.backward}`);
        w.writeLine(`* \`\`\``);
        w.writeLine(`*`);
    }
    if (includeParamsAndReturn) {
        if (inputName !== "this") {
            w.writeLine(`* @param ${inputName} the input tensor of any shape`);
        }
        if (isBinary) {
            w.writeLine(`* @param other the other tensor whose shape is broadcastable with the input tensor`);
            if (hasAlpha) {
                w.writeLine(`* @param alpha the alpha value to multiply \`other\` with`);
            }
            else {
            }
        }
        else {
            if (hasAlpha) {
                w.writeLine(`* @param alpha the alpha value`);
            }
            else {
            }
        }
        w.writeLine(`* @returns the output tensor`);
    }
    w.writeLine(`*/`);
}

// Write global ops
function writeOpsCode(): void {
    const w = new CodeWriter();
    w.writeLine(`import * as functions from "./functions_opgen";
import { Tensor } from "./tensor";
import { unary, unaryWithAlpha, binary, binaryWithAlpha, reduction } from "./ops_high";`);
    for (const [opSpec, kernelSpec] of opKernelSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        if (isInplace) {
            continue;
        }
        const isGrad = kernelSpec.name.endsWith("_grad");
        if (isGrad) continue;
        const isStrided = kernelSpec.name.includes("_strided");
        if (isStrided) continue;
        const isOtherScalar = kernelSpec.name.includes("_scalar");
        if (isOtherScalar) continue;
        const isBinary = opSpec.type === "binary";
        const isReduction = opSpec.type === "reduction";
        if (isReduction && kernelSpec.name.endsWith("_dim")) continue;
        const hasAlpha = opSpec.alpha ?? false;
        const funcName = kernelSpec.name[0].toUpperCase() + kernelSpec.name.slice(1) + "Function";
        const writeHeader = (name: string, isAlias: boolean) => {
            writeOpDocs(opSpec, "input", isAlias, w);
            if (isReduction) {
                w.writeLine(`export function ${name}(input: Tensor, dim?: number | number[], keepdim?: boolean): Tensor {`);
            } else if (isBinary) {
                if (hasAlpha) {
                    w.writeLine(`export function ${name}(input: Tensor, other: number | Tensor, alpha?: number): Tensor {`);
                }
                else {
                    w.writeLine(`export function ${name}(input: Tensor, other: number | Tensor): Tensor {`);
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
        writeHeader(kernelSpec.name, false);
        w.indent();
        if (isReduction) {
            w.writeLine(`return reduction(functions.${funcName}, input, dim, keepdim);`);
        }
        else if (isBinary) {
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
            writeHeader(alias, true);
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
    writeFile(path, code);
}
writeOpsCode();

// Write autograd functions
function writeNNModulesCode(): void {
    const w = new CodeWriter();
    w.writeLine(`import { Tensor } from "./tensor";
import { Module } from "./nn_module";`);
    for (const opSpec of opRegistry) {
        if (!opSpec.nnOp)
            continue;
        const name = opSpec.name;
        const isBinary = opSpec.type === "binary";
        const isReduction = opSpec.type === "reduction";
        const hasAlpha = opSpec.alpha ?? false;
        const nnName = opSpec.nnName ?? (name[0].toUpperCase() + name.slice(1));
        const params: string[] = ["input: Tensor"];
        const args: string[] = [];
        if (isBinary) {
            params.push("other: Tensor");
            args.push("other");
        }
        if (hasAlpha) {
            params.push("alpha?: number");
            args.push("alpha");
        }
        const paramsStr = params.join(", ");
        const argsStr = args.join(", ");
        writeOpDocs(opSpec, "input", false, w, false);
        w.writeLine(`export class ${nnName} extends Module {`);
        w.indent();

        // Forward
        w.writeLine(`forward(${paramsStr}): Tensor {`);
        w.indent();
        w.writeLine(`return input.${opSpec.name}(${argsStr});`);
        w.dedent();
        w.writeLine(`}`);

        // End Module
        w.dedent();
        w.writeLine(`}`);
    }
    w.writeLine("");
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/nn_opgen.ts";
    writeFile(path, code);
}
writeNNModulesCode();

