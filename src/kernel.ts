import { Device } from "./device";
import { ATypedArray, Dtype } from "./dtype";
import { ExprCode, evalCode, compileCode, CompiledExpr, EvalEnv } from "./expr";
import { CodeWriter } from "./opgen";

export type KernelParamType = "u32" | "f32";
export type KernelParamValue = number;

export type ShaderType =
    | KernelParamType
    | "u8"
    | "array<u8>"
    | "i32"
    | "array<i32>"
    | "array<u32>"
    | "array<f32>";

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
    shaderType: KernelParamType;
}

export interface KernelConfigSpec {
    name: string;
}

export type KernelConfigValue = string | number;
export type KernelConfigInput = { [name: string]: KernelConfigValue };
export type KernelParamsInput = { [name: string]: KernelParamValue };
export type KernelConfig = KernelConfigValue[];
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

    getRunEnv(parameters: KernelParamsInput): EvalEnv {
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
                `Missing config value for ${configSpec.name} in kernel ${spec.name}`
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

export function getKernelShaderCode(
    spec: KernelSpec,
    config: KernelConfig
): string {
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
    const env: { [name: string]: any } = {};
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[i];
        env[configSpec.name] = configValue;
    }
    const workgroupSizeX = Math.ceil(evalCode(spec.workgroupSize[0], env));
    const workgroupSizeY = Math.ceil(evalCode(spec.workgroupSize[1], env));
    const workgroupSizeZ = Math.ceil(evalCode(spec.workgroupSize[2], env));
    shaderCodeParts.push(
        `@compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}, ${workgroupSizeZ})`
    );
    shaderCodeParts.push(
        `fn main(@builtin(global_invocation_id) global_id : vec3u) {`
    );
    shaderCodeParts.push("    " + spec.shader.trim());
    shaderCodeParts.push("}");
    return shaderCodeParts.join("\n");
}

const javaScriptSubstitutions: [RegExp, string][] = [
    ["global_id\\.x", "global_id_x"],
    ["global_id\\.y", "global_id_y"],
    ["global_id\\.z", "global_id_z"],
].map(([regex, replacement]) => [new RegExp(regex, "g"), replacement]);

export function getKernelJavaScriptCode(
    spec: KernelSpec,
    config: KernelConfig
): string {
    const env: { [name: string]: any } = {};
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[i];
        env[configSpec.name] = configValue;
    }

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
    w.writeLine(`const workgroupSizeX = ${workgroupSizeX};`);
    w.writeLine(`const workgroupSizeY = ${workgroupSizeY};`);
    w.writeLine(`const workgroupSizeZ = ${workgroupSizeZ};`);
    w.writeLine(`console.log("workgroupCount", workgroupCountX, workgroupCountY, workgroupCountZ);`);
    let jsCode = spec.shader.trim();
    for (const [regex, replacement] of javaScriptSubstitutions) {
        jsCode = jsCode.replace(regex, replacement);
    }
    w.writeLine(jsCode);
    w.dedent();
    w.writeLine(`})`);
    return w.toString();
}
