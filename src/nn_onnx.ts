import { Module } from "./nn_module";
import { onnx } from "./onnx";
import { fetch } from "cross-fetch";
import { Tensor, TensorSpec } from "./tensor";
import { gather, matmul, tensor } from "./ops_artisanal";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";
import { cpuDevice, getDevice } from "./devices";
import Long from "long";
import { Dtype, newTypedArrayForDtype } from "./dtype";
import {
    flatDataToArray,
    ArrayBufferStorage,
    TensorArrayData,
} from "./storage";
import type { Device } from "./device";

export class ONNXModule extends Module {
    readonly inputs: onnx.IValueInfoProto[];
    readonly outputs: onnx.IValueInfoProto[];
    readonly nodes: onnx.INodeProto[];
    readonly nodeFromOutput: {
        [outputName: string]: { node: onnx.INodeProto; outputIndex: number };
    } = {};
    readonly tensorSpecNameToBufferName: { [tensorSpecName: string]: string } =
        {};
    constructor(model: onnx.IModelProto) {
        super();
        const graph = model.graph;
        if (!graph) {
            throw new Error("Model does not contain a graph");
        }
        const inputs = graph.input;
        if (!inputs) {
            throw new Error("Model graph does not contain inputs");
        }
        const outputs = graph.output;
        if (!outputs) {
            throw new Error("Model graph does not contain outputs");
        }
        const nodes = graph.node;
        if (!nodes) {
            throw new Error("Model graph does not contain nodes");
        }
        this.inputs = inputs;
        this.outputs = outputs;
        this.nodes = nodes;
        for (const node of nodes) {
            for (const [outputIndex, output] of (node.output || []).entries()) {
                this.nodeFromOutput[output] = { node, outputIndex };
            }
        }
        // Create buffers for initialized data
        const device = getDevice();
        for (const init of graph.initializer || []) {
            const name = init.name || "";
            if (name.length === 0) {
                throw new Error("Initializer does not have a name");
            }
            const bufferName = name.replace(/[^a-zA-Z0-9_]/g, "_");
            this.tensorSpecNameToBufferName[name] = bufferName;
            this.registerBuffer(bufferName, initToTensor(init, device));
        }
        // Create constant
        for (const node of nodes) {
            if (node.opType === "Constant") {
                const attr = node.attribute?.find(
                    (a) => a.name === "value"
                );
                if (!attr) {
                    throw new Error("Constant does not have a value");
                }
                if (attr.t) {
                    const name = (node.output || [])[0] || "";
                    if (name.length === 0) {
                        throw new Error("Constant does not have a name");
                    }
                    const bufferName = name.replace(/[^a-zA-Z0-9_]/g, "_");
                    this.tensorSpecNameToBufferName[name] = bufferName;
                    const ctensor = initToTensor(attr.t, cpuDevice);
                    this.registerBuffer(bufferName, ctensor);
                } else {
                    console.warn(`Constant ${node.name} has no tensor value`);
                }
            }
        }
    }
    forward(inputs: Tensor[]): Tensor[] {
        const outputs: Tensor[] = [];
        const env: { [name: string]: ONNXValue } = {};
        for (const [i, input] of inputs.entries()) {
            const inputName = this.inputs[i].name || "";
            if (inputName.length === 0) {
                throw new Error(`Input ${i} does not have a name`);
            }
            env[inputName] = input;
        }
        for (const [i, output] of this.outputs.entries()) {
            const outputName = output.name || "";
            if (outputName.length === 0) {
                throw new Error(`Output ${i} does not have a name`);
            }
            const outputT = this.evaluateTensor(outputName, env);
            if (!(outputT instanceof Tensor)) {
                throw new Error(`Output ${i} is not a tensor`);
            }
            outputs.push(outputT);
        }
        return outputs;
    }
    private evaluateTensor(
        tensorSpecName: string,
        env: { [name: string]: ONNXValue }
    ): ONNXValue {
        // Has it already been evaluated?
        if (tensorSpecName in env) {
            return env[tensorSpecName];
        }
        // Has it been initialized?
        if (tensorSpecName in this.tensorSpecNameToBufferName) {
            const bufferName = this.tensorSpecNameToBufferName[tensorSpecName];
            const buffer = this.getBuffer(bufferName);
            if (!buffer) {
                throw new Error(`Buffer ${bufferName} not found`);
            }
            env[tensorSpecName] = buffer;
            return buffer;
        }
        // Is it a node?
        const node = this.nodeFromOutput[tensorSpecName];
        if (!node) {
            throw new Error(`Tensor ${tensorSpecName} does not have a node`);
        }
        // Evaluate inputs
        const inputs: ONNXValue[] = [];
        for (const tensorSpecName of node.node.input || []) {
            inputs.push(this.evaluateTensor(tensorSpecName, env));
        }
        // Evaluate node
        const nodeOutputNames = node.node.output || [];
        const outputs = evalNode(node.node, inputs);
        if (outputs.length !== nodeOutputNames.length) {
            throw new Error(
                `Node ${node.node.name} has ${nodeOutputNames.length} outputs but evaluated to ${outputs.length} outputs`
            );
        }
        for (const [i, output] of outputs.entries()) {
            const outputName = nodeOutputNames[i];
            env[outputName] = output;
        }
        const output = outputs[node.outputIndex];
        env[tensorSpecName] = output;
        return output;
    }
    static async loadUrlAsync(url: string): Promise<ONNXModule> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(
                `HTTP request for model failed. Status: ${response.status}`
            );
        }
        const buffer = await response.arrayBuffer();
        const model = onnx.ModelProto.decode(new Uint8Array(buffer));
        return new ONNXModule(model);
    }
}

function longDimsToShape(dims: (Long | number)[]): Shape {
    const shape: number[] = [];
    for (const rawdim of dims) {
        if (rawdim instanceof Long) {
            const dim = rawdim.toNumber();
            if (dim < 0) {
                throw new Error("ONNX tensor dimension is negative");
            }
            shape.push(dim);
        } else {
            const dim = rawdim;
            if (dim < 0) {
                throw new Error("ONNX tensor dimension is negative");
            }
            shape.push(dim);
        }
    }
    return shape;
}

function onnxDataTypeToDType(onnxDataType: onnx.TensorProto.DataType): Dtype {
    switch (onnxDataType) {
        case onnx.TensorProto.DataType.INT8:
            return "int8";
        case onnx.TensorProto.DataType.FLOAT:
            return "float32";
        case onnx.TensorProto.DataType.INT64:
            return "int64";
        default:
            throw new Error(`ONNX data type ${onnxDataType} not supported`);
    }
}

function initToTensor(init: onnx.ITensorProto, device: Device): Tensor {
    const shape: Shape = longDimsToShape(init.dims || []);
    const strides = defaultStrides(shape);
    let dtype = onnxDataTypeToDType(init.dataType || 1);
    let data: number[];
    if (dtype === "int64") {
        const oldDtype = dtype;
        if (init.rawData && init.rawData.length > 0) {
            let bytes = init.rawData;
            let byteOffset = bytes.byteOffset;
            const length = init.rawData.byteLength / 8;
            if (byteOffset % 8 !== 0) {
                const newBuffer = new ArrayBuffer(bytes.byteLength);
                const newBytes = new Uint8Array(newBuffer);
                newBytes.set(bytes);
                bytes = newBytes;
                byteOffset = bytes.byteOffset;
            }
            const in64 = new BigInt64Array(bytes.buffer, byteOffset, length);
            const minInt32 = -2147483648n;
            const maxInt32 = 2147483647n;
            data = Array.from(in64).map((x) => {
                if (x < minInt32) {
                    console.warn(`Clipping int64 value ${x} to -2147483648`);
                    return -2147483648;
                }
                if (x > maxInt32) {
                    console.warn(`Clipping int64 value ${x} to 2147483647`);
                    return 2147483647;
                }
                return Number(x);
            });
            dtype = "int32";
        } else if (init.int64Data && init.int64Data.length > 0) {
            data = init.int64Data.map((x) =>
                x instanceof Long ? x.toNumber() : (x as number)
            );
            dtype = "int32";
        } else if (init.uint64Data && init.uint64Data.length > 0) {
            data = init.uint64Data.map((x) =>
                x instanceof Long ? x.toNumber() : (x as number)
            );
            dtype = "uint32";
        } else {
            throw new Error("Initializer for int64 data cannot be found");
        }
    } else {
        if (init.rawData && init.rawData.length > 0) {
            data = Array.from(init.rawData);
        } else if (init.floatData && init.floatData.length > 0) {
            data = init.floatData;
        } else if (init.int32Data && init.int32Data.length > 0) {
            data = init.int32Data;
        } else {
            throw new Error("Initializer does not contain data");
        }
    }
    const spec: TensorSpec = {
        data,
        shape,
        strides,
        dtype,
        device,
    };
    return new Tensor(spec);
}

function attrNumbers(node: onnx.INodeProto, name: string): number[] {
    const attr = node.attribute?.find((x) => x.name === name);
    if (!attr) {
        throw new Error(
            `Attribute ${name} of node ${node.name} (${node.opType}) not found`
        );
    }
    const atype = attr.type;
    switch (atype) {
        case onnx.AttributeProto.AttributeType.INT: {
            const v = attr.i!;
            if (typeof v == "number") {
                return [v];
            } else {
                return [v.toNumber()];
            }
        }
        case onnx.AttributeProto.AttributeType.INTS: {
            const v = attr.ints!;
            return v.map((x) => (typeof x == "number" ? x : x.toNumber()));
        }
        case onnx.AttributeProto.AttributeType.FLOAT: {
            return [attr.f!];
        }
        default:
            throw new Error(
                `Attribute ${name} of node ${node.name} (${node.opType}) is not a number`
            );
    }
}

type StructuredArray = {
    data: TensorArrayData | number;
    dtype: Dtype;
    shape: Shape;
};

type ONNXValue = Tensor | StructuredArray;

function valueToTensor(value: ONNXValue): Tensor {
    if (value instanceof Tensor) {
        return value;
    } else {
        return tensor(value);
    }
}

function valueToArray(value: ONNXValue): StructuredArray {
    if (value instanceof Tensor) {
        const storage = value.storage;
        if (storage.device.type !== "cpu") {
            throw new Error("GPU tensor cannot be converted to array");
        }
        const data = (storage as ArrayBufferStorage).getTypedArray(value.dtype);
        return {
            data: flatDataToArray(data, value.shape, value.strides),
            dtype: value.dtype,
            shape: value.shape,
        };
    } else {
        return value;
    }
}

function onnxGather(
    data: StructuredArray,
    axis: number,
    indices: StructuredArray
): StructuredArray {
    const r = data.shape.length;
    const q = indices.shape.length;
    const outputRank = q + (r - 1);
    axis = axis < 0 ? r + axis : axis;
    const outputShape = new Array(outputRank);
    const outputStrides = new Array(outputRank);
    let j = 0;
    for (let i = 0; i < outputRank; i++) {
        if (i < axis) {
            outputShape[i] = data.shape[i];
        } else if (i < axis + q) {
            outputShape[i] = indices.shape[j];
            j++;
        } else {
            outputShape[i] = data.shape[i - q + 1];
        }
    }
    const outputSize = outputShape.reduce((a, b) => a * b, 1);
    let outputData: TensorArrayData | number;
    if (r == 1 && outputRank === 0) {
        const inputData = data.data as number[];
        const indicesData = indices.data as number;
        let output: number = 0;
        const k = indicesData;
        const inputIndex0 = k;
        output = inputData[inputIndex0];
        outputData = output;
    } else if (r == 1 && outputRank === 1) {
        const inputData = data.data as number[];
        const indicesData = indices.data as number[];
        let flatDataIndex = 0;
        const flatData: number[] = new Array(outputSize);
        for (let o0 = 0; o0 < outputShape[0]; o0++) {
            // output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k, j_{0}, ..., j_{r-2}]
            const k = indicesData[o0];
            const inputIndex0 = k;
            flatData[flatDataIndex] = inputData[inputIndex0];
            flatDataIndex++;
        }
        outputData = flatData;
    } else {
        throw new Error(`agather rank ${outputRank} not supported`);
    }
    return {
        data: outputData,
        dtype: data.dtype,
        shape: outputShape,
    };
}

function onnxConcat(inputs: StructuredArray[], axis: number): StructuredArray {
    const r = inputs[0].shape.length;
    axis = axis < 0 ? r + axis : axis;
    const outputShape = inputs[0].shape.slice();
    outputShape[axis] = inputs.reduce((a, x) => a + x.shape[axis], 0);
    const outputSize = shapeSize(outputShape);
    let outputData: TensorArrayData;
    if (outputShape.length === 1) {
        const output = new Array<number>(outputSize);
        let flatDataIndex = 0;
        for (const input of inputs) {
            const inputData = input.data as number[];
            for (let i0 = 0; i0 < input.shape[0]; i0++) {
                output[flatDataIndex] = inputData[i0];
                flatDataIndex++;
            }
        }
        outputData = output;
    }
    else {
        throw new Error(`concat rank ${outputShape.length} not supported`);
    }
    return {
        data: outputData,
        dtype: inputs[0].dtype,
        shape: outputShape,
    };
}

function onnxUnsqueeze(data: StructuredArray, axes: number[]): StructuredArray {
    const r = data.shape.length;
    const outputShape = new Array(r + axes.length);
    let i = 0;
    let j = 0;
    for (let k = 0; k < outputShape.length; k++) {
        if (axes[j] === k) {
            outputShape[k] = 1;
            j++;
        } else {
            outputShape[k] = data.shape[i];
            i++;
        }
    }
    let outputData: TensorArrayData;
    if (r === 0 && outputShape.length === 1) {
        const inputData = data.data as number;
        const output = new Array<number>(1);
        output[0] = inputData;
        outputData = output;
    } else if (r === 1 && outputShape.length === 2) {
        const inputData = data.data as number[];
        const n0 = outputShape[0];
        const output = new Array<number[]>(n0);
        for (let i0 = 0; i0 < n0; i0++) {
            const n1 = outputShape[1];
            output[i0] = new Array<number>(n1);
            for (let i1 = 0; i1 < n1; i1++) {
                const inputIndex = axes[0] === 0 ? i1 : i0;
                output[i0][i1] = inputData[inputIndex];
            }
        }
        outputData = output;
    }
    else {
        throw new Error(`ONNX Unsqueeze rank ${outputShape.length} not supported`);
    }
    return {
        data: outputData,
        dtype: data.dtype,
        shape: outputShape,
    };
}

function evalNode(node: onnx.INodeProto, inputs: ONNXValue[]): ONNXValue[] {
    switch (node.opType) {
        case "Concat": {
            const axis = attrNumbers(node, "axis")[0];
            return [onnxConcat(inputs.map(valueToArray), axis)];
        }
        case "Constant": {
            throw new Error("Constant nodes should be removed");
        }
        case "Gather": {
            const dim = attrNumbers(node, "axis")[0];
            if (inputs[0] instanceof Tensor) {
                return [gather(inputs[0], dim, valueToTensor(inputs[1]))];
            } else {
                return [onnxGather(inputs[0], dim, valueToArray(inputs[1]))];
            }
        }
        case "MatMul": {
            return [matmul(valueToTensor(inputs[0]), valueToTensor(inputs[1]))];
        }
        case "Mul": {
            return [valueToTensor(inputs[0]).mul(valueToTensor(inputs[1]))];
        }
        case "Reshape": {
            const input = valueToTensor(inputs[0]);
            const shape = valueToArray(inputs[1]).data as number[];
            return [input.reshape(shape)];
        }
        case "Shape": {
            return [
                {
                    data: inputs[0].shape,
                    dtype: "int32",
                    shape: [inputs[0].shape.length],
                },
            ];
        }
        case "Unsqueeze": {
            const axes = attrNumbers(node, "axes");
            return [onnxUnsqueeze(valueToArray(inputs[0]), axes)];
        }
        default:
            throw new Error(`Cannot evaluate ONNX node type '${node.opType}'`);
    }
}
