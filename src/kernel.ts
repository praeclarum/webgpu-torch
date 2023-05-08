import { Device } from "./device";
import { ATypedArray, Dtype } from "./dtype";
import { ExprCode, evalCode, compileCode, CompiledExpr, EvalEnv } from "./expr";
import { CodeWriter } from "./opgen";

export type ShaderType =
    | "u8"
    | "array<u8>"
    | "i32"
    | "array<i32>"
    | "u32"
    | "array<u32>"
    | "f32"
    | "array<f32>"
    | "vec3<i32>"
    | "vec3<u32>"
    | "vec3<f32>";

export interface KernelSpec {
    name: string;
    config: KernelConfigSpec[];
    workgroupSize: [ExprCode, ExprCode, ExprCode];
    parameters: KernelParamSpec[];
    workgroupCount: [ExprCode, ExprCode, ExprCode];
    inputs: KernelInputSpec[];
    outputs: KernelOutputSpec[];
    shader: string;
}

export interface KernelInputSpec {
    name: string;
    shaderType: ShaderType;
}

export interface KernelOutputSpec {
    name: string;
    shaderType: ShaderType;
    size: ExprCode;
}

export interface KernelParamSpec {
    name: string;
    shaderType: "u32" | "f32";
}

export interface KernelConfigSpec {
    name: string;
}

export type KernelConfigInput = { [name: string]: string | number };
export type KernelParamsInput = { [name: string]: number };
export type KernelConfig = (string | number)[];
export type KernelKey = string;

export abstract class Kernel {
    private _key: KernelKey;
    private _spec: KernelSpec;
    private _config: KernelConfig;
    private _device: Device;
    private _workgroupCountXFunc: CompiledExpr;
    private _workgroupCountYFunc: CompiledExpr;
    private _workgroupCountZFunc: CompiledExpr;
    protected _outputSizeFuncs: CompiledExpr[];
    get key(): KernelKey {
        return this._key;
    }
    get spec(): KernelSpec {
        return this._spec;
    }
    get config(): KernelConfig {
        return this._config;
    }
    get device(): Device {
        return this._device;
    }
    constructor(spec: KernelSpec, config: KernelConfig, device: Device) {
        this._key = getKernelKey(spec, config);
        this._device = device;
        this._spec = spec;
        this._config = config;
        this._workgroupCountXFunc = compileCode(spec.workgroupCount[0]);
        this._workgroupCountYFunc = compileCode(spec.workgroupCount[1]);
        this._workgroupCountZFunc = compileCode(spec.workgroupCount[2]);
        this._outputSizeFuncs = [];
        for (let i = 0; i < this._spec.outputs.length; i++) {
            const outputSpec = this._spec.outputs[i];
            const outputElementCount = compileCode(outputSpec.size);
            this._outputSizeFuncs.push(outputElementCount);
        }
    }

    abstract run(
        inputs: (GPUBuffer | ATypedArray)[],
        parameters: KernelParamsInput,
        outputs?: (GPUBuffer | ATypedArray)[]
    ): (GPUBuffer | ATypedArray)[];

    protected getRunEnv(parameters: KernelParamsInput): EvalEnv {
        const env: EvalEnv = {};
        for (let i = 0; i < this._spec.config.length; i++) {
            const configSpec = this._spec.config[i];
            const configValue = this._config[i];
            env[configSpec.name] = configValue;
        }
        for (let i = 0; i < this.spec.parameters.length; i++) {
            const param = this.spec.parameters[i];
            const paramValue = parameters[param.name];
            if (paramValue === undefined) {
                throw new Error(`Missing parameter ${param.name}`);
            }
            env[param.name] = paramValue;
        }
        return env;
    }

    getWorkgroupCounts(env: EvalEnv): [number, number, number] {
        return [
            Math.ceil(this._workgroupCountXFunc(env)),
            Math.ceil(this._workgroupCountYFunc(env)),
            Math.ceil(this._workgroupCountZFunc(env)),
        ];
    }
}

export function getKernelConfig(
    spec: KernelSpec,
    config: KernelConfigInput
): KernelConfig {
    let configValues: KernelConfig = [];
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[configSpec.name];
        if (configValue === undefined) {
            throw new Error(
                `Missing config value "${configSpec.name}" for kernel "${spec.name}"`
            );
        }
        configValues.push(configValue);
    }
    return configValues;
}

export function getKernelKey(
    spec: KernelSpec,
    config: KernelConfig
): KernelKey {
    let keyParts: string[] = [spec.name];
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[i];
        keyParts.push(`${configSpec.name}=${configValue}`);
    }
    return keyParts.join(",");
}

export function getShaderTypeElementByteSize(shaderType: ShaderType): number {
    switch (shaderType) {
        case "f32":
        case "i32":
        case "u32":
        case "array<f32>":
        case "array<i32>":
        case "array<u32>":
            return 4;
        case "u8":
        case "array<u8>":
            return 1;
        default:
            throw new Error(`Unknown shader type ${shaderType}`);
    }
}

export function shaderTypeToDtype(shaderType: ShaderType): Dtype {
    switch (shaderType) {
        case "f32":
        case "array<f32>":
            return "float32";
        case "i32":
        case "array<i32>":
            return "int32";
        case "u32":
        case "array<u32>":
            return "uint32";
        case "u8":
        case "array<u8>":
            return "uint8";
        default:
            throw new Error(`Unknown shader type ${shaderType}`);
    }
}

function getIdentifiers(code: string): string[] {
    const identifierRegex = /[a-zA-Z_][a-zA-Z0-9_]*/g;
    const identifiers = new Set<string>();
    let match: RegExpExecArray | null;
    while ((match = identifierRegex.exec(code)) !== null) {
        identifiers.add(match[0]);
    }
    return Array.from(identifiers);
}

function configShader(
    spec: KernelSpec,
    config: KernelConfig
): [string, EvalEnv] {
    const substituions: [string, string][] = [];
    const env: EvalEnv = {};
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[i];
        substituions.push([configSpec.name, configValue.toString()]);
        env[configSpec.name] = configValue;
    }
    let result = spec.shader.trim();
    for (let [key, value] of substituions) {
        result = result.replace(new RegExp(`\\$\\$${key}\\$\\$`, "g"), value);
    }
    return [result, env];
}

export function getKernelShaderCode(
    spec: KernelSpec,
    config: KernelConfig
): string {
    const [configdShader, env] = configShader(spec, config);

    let shaderCodeParts: string[] = ["// " + spec.name + " kernel"];
    shaderCodeParts.push(`struct ${spec.name}Parameters {`);
    for (let i = 0; i < spec.parameters.length; i++) {
        let parameter = spec.parameters[i];
        shaderCodeParts.push(`    ${parameter.name}: ${parameter.shaderType},`);
    }
    shaderCodeParts.push(`}`);
    let bindingIndex = 0;
    for (let i = 0; i < spec.inputs.length; i++, bindingIndex++) {
        let input = spec.inputs[i];
        shaderCodeParts.push(
            `@group(0) @binding(${bindingIndex}) var<storage, read> ${input.name}: ${input.shaderType};`
        );
    }
    for (let i = 0; i < spec.outputs.length; i++, bindingIndex++) {
        let output = spec.outputs[i];
        shaderCodeParts.push(
            `@group(0) @binding(${bindingIndex}) var<storage, read_write> ${output.name}: ${output.shaderType};`
        );
    }
    shaderCodeParts.push(
        `@group(0) @binding(${bindingIndex}) var<storage, read> parameters: ${spec.name}Parameters;`
    );
    const workgroupSizeX = Math.ceil(evalCode(spec.workgroupSize[0], env));
    const workgroupSizeY = Math.ceil(evalCode(spec.workgroupSize[1], env));
    const workgroupSizeZ = Math.ceil(evalCode(spec.workgroupSize[2], env));
    shaderCodeParts.push(
        `@compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}, ${workgroupSizeZ})`
    );
    shaderCodeParts.push(`fn main(`);
    let head = "";
    if (configdShader.includes("global_id")) {
        shaderCodeParts.push(
            `    ${head}@builtin(global_invocation_id) global_id: vec3u`
        );
        head = ", ";
    }
    if (configdShader.includes("local_id")) {
        shaderCodeParts.push(
            `    ${head}@builtin(local_invocation_id) local_id: vec3u`
        );
        head = ", ";
    }
    shaderCodeParts.push(`    ) {`);
    shaderCodeParts.push("    " + configdShader);
    shaderCodeParts.push("}");
    const shaderCode = shaderCodeParts.join("\n");
    // console.log(shaderCode);
    return shaderCode;
}

const javaScriptSubstitutions: [RegExp, string][] = [
    ["(\\d+)u", "$1"],
    ["(\\d+)f", "$1"],
    ["(\\d+\\.\\d+)f", "$1"],
    ["global_id\\.x", "global_id_x"],
    ["global_id\\.y", "global_id_y"],
    ["global_id\\.z", "global_id_z"],
    ["local_id\\.x", "local_id_x"],
    ["local_id\\.y", "local_id_y"],
    ["local_id\\.z", "local_id_z"],
    ["workgroupBarrier\\s*\\(\\s*\\)\\s*;", "yield \"workgroupBarrier\";"],
    ["storageBarrier\\s*\\(\\s*\\)\\s*;", "yield \"storageBarrier\";"],
].map(([regex, replacement]) => [new RegExp(regex, "g"), replacement]);

// Add all the Math. functions to the substitution list
for (const name of Object.getOwnPropertyNames(Math)) {
    if (typeof (Math as any)[name] === "function") {
        javaScriptSubstitutions.push([new RegExp(`\\b${name}\\(`, "g"), `Math.${name}(`]);
    }
}

const javaScriptGlobalFunctions: { [name: string]: string } = {
    exp2: "function exp2(x) { return Math.pow(2, x); }",
    fract: "function fract(x) { return x - Math.floor(x); }",
    select: "function select(falseValue, trueValue, condition) { return condition ? trueValue : falseValue; }",
};

export function getKernelJavaScriptCode(
    spec: KernelSpec,
    config: KernelConfig
): string {
    const [configdShader, env] = configShader(spec, config);

    const usesGlobalId = configdShader.includes("global_id");
    const usesLocalId = configdShader.includes("local_id");
    const usesBarrier =
        configdShader.includes("workgroupBarrier") ||
        configdShader.includes("storageBarrier");

    // Build up the body of the kernel function
    let jsCode = configdShader;
    for (const [regex, replacement] of javaScriptSubstitutions) {
        jsCode = jsCode.replace(regex, replacement);
    }

    // Find needed functions
    const identifiers = getIdentifiers(jsCode);
    const neededFunctions = new Set<string>();
    for (const identifier of identifiers) {
        if (identifier in javaScriptGlobalFunctions) {
            neededFunctions.add(identifier);
        }
    }
    const neededFunctionsArray = Array.from(neededFunctions);

    // Write the whole function
    const w = new CodeWriter();
    const params: string[] = [];
    let bindingIndex = 0;
    for (let i = 0; i < spec.inputs.length; i++, bindingIndex++) {
        let input = spec.inputs[i];
        params.push(input.name);
    }
    for (let i = 0; i < spec.outputs.length; i++, bindingIndex++) {
        let output = spec.outputs[i];
        params.push(output.name);
    }
    params.push("parameters");
    params.push("workgroupCountX");
    params.push("workgroupCountY");
    params.push("workgroupCountZ");
    const workgroupSizeX = Math.ceil(evalCode(spec.workgroupSize[0], env));
    const workgroupSizeY = Math.ceil(evalCode(spec.workgroupSize[1], env));
    const workgroupSizeZ = Math.ceil(evalCode(spec.workgroupSize[2], env));
    w.writeLine(`((${params.join(", ")}) => {`);
    w.indent();

    // Write dependent functions
    for (const neededFunction of neededFunctionsArray) {
        w.writeLine(javaScriptGlobalFunctions[neededFunction]);
    }

    // Write the kernel function
    const kernelParams = [];
    if (usesGlobalId) {
        kernelParams.push("global_id_x");
        kernelParams.push("global_id_y");
        kernelParams.push("global_id_z");
    }
    if (usesLocalId) {
        kernelParams.push("local_id_x");
        kernelParams.push("local_id_y");
        kernelParams.push("local_id_z");
    }
    const kernelParamsString = kernelParams.join(", ");
    w.writeLine(
        `function${usesBarrier ? "*" : ""} ${
            spec.name
        }Kernel(${kernelParamsString}) {`
    );
    w.indent();
    w.writeLine(jsCode);
    w.dedent();
    w.writeLine(`}`);
    // for (let p of params) {
    //     w.writeLine(`console.log("param", "${p}", typeof ${p}, ${p});`);
    // }
    const workgroupSize = workgroupSizeX * workgroupSizeY * workgroupSizeZ;
    if (usesBarrier) {
        w.writeLine(`var barriers = new Array(${workgroupSize});`);
    }
    w.writeLine(`for (let wgZ = 0; wgZ < workgroupCountZ; wgZ++) {`);
    w.indent();
    w.writeLine(`for (let wgY = 0; wgY < workgroupCountY; wgY++) {`);
    w.indent();
    w.writeLine(
        `for (let group_id_x = 0; group_id_x < workgroupCountX; group_id_x++) {`
    );
    w.indent();
    w.writeLine(`const globalStartX = group_id_x * ${workgroupSizeX};`);
    w.writeLine(`const globalEndX = globalStartX + ${workgroupSizeX};`);
    w.writeLine(`const globalStartY = wgY * ${workgroupSizeY};`);
    w.writeLine(`const globalEndY = globalStartY + ${workgroupSizeY};`);
    w.writeLine(`const globalStartZ = wgZ * ${workgroupSizeZ};`);
    w.writeLine(`const globalEndZ = globalStartZ + ${workgroupSizeZ};`);
    if (usesBarrier) {
        w.writeLine(`let allDone = true;`);
    }
    w.writeLine(
        `for (let global_id_z = globalStartZ; global_id_z < globalEndZ; global_id_z++) {`
    );
    w.indent();
    w.writeLine(
        `for (let global_id_y = globalStartY; global_id_y < globalEndY; global_id_y++) {`
    );
    w.indent();
    w.writeLine(
        `for (let global_id_x = globalStartX; global_id_x < globalEndX; global_id_x++) {`
    );
    w.indent();
    if (usesLocalId) {
        w.writeLine(`const local_id_x = global_id_x - globalStartX;`);
        w.writeLine(`const local_id_y = global_id_y - globalStartY;`);
        w.writeLine(`const local_id_z = global_id_z - globalStartZ;`);
    }
    if (usesBarrier) {
        w.writeLine(
            `const local_index = local_id_x + local_id_y * ${workgroupSizeX} + local_id_z * ${
                workgroupSizeX * workgroupSizeY
            };`
        );
        w.writeLine(
            `const barrier = ${spec.name}Kernel(${kernelParamsString});`
        );
        w.writeLine(`const firstBarrierValue = barrier.next();`);
        w.writeLine(`allDone = allDone && firstBarrierValue.done;`);
        w.writeLine(`barriers[local_index] = barrier;`);
    } else {
        w.writeLine(`${spec.name}Kernel(${kernelParamsString});`);
    }
    w.dedent();
    w.writeLine(`}`);
    w.dedent();
    w.writeLine(`}`);
    w.dedent();
    w.writeLine(`}`);
    if (usesBarrier) {
        w.writeLine(`while (!allDone) {`);
        w.indent();
        w.writeLine(`allDone = true;`);
        w.writeLine(`for (let local_index = 0; local_index < ${workgroupSize}; local_index++) {`);
        w.indent();
        w.writeLine(`const barrier = barriers[local_index];`);
        w.writeLine(`const barrierValue = barrier.next();`);
        w.writeLine(`allDone = allDone && barrierValue.done;`);
        w.dedent();
        w.writeLine(`}`);
        w.dedent();
        w.writeLine(`}`);
    }
    w.dedent();
    w.writeLine(`}`);
    w.dedent();
    w.writeLine(`}`);
    w.dedent();
    w.writeLine(`}`);
    w.dedent();
    w.writeLine(`})`);
    const code = w.toString();
    // console.log(code);
    return code;
}
