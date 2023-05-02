import { ATypedArray, Dtype, dtypeArrayCtors } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { TensorArrayData } from "./tensor_if";

export abstract class UntypedStorage {
    abstract get buffer(): ArrayBuffer | null;
    tryGetTypedArray(dtype: Dtype): ATypedArray | null {
        const buffer = this.buffer;
        if (buffer === null) {
            return null;
        }
        if (dtype in dtypeArrayCtors) {
            return new dtypeArrayCtors[dtype](buffer);
        }
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
    getTypedArray(dtype: Dtype): ATypedArray {
        const result = this.tryGetTypedArray(dtype);
        if (result !== null) {
            return result;
        }
        throw new Error("Storage is not mapped");
    }
}

export class ArrayBufferStorage extends UntypedStorage {
    private _buffer: ArrayBuffer;
    constructor(byteSize: number | ArrayBuffer | ATypedArray) {
        super();
        if (typeof byteSize === "number") {
            this._buffer = new ArrayBuffer(byteSize as number);
        }
        else if (byteSize instanceof ArrayBuffer) {
            this._buffer = byteSize;
        }
        else if (byteSize instanceof Uint8Array) {
            this._buffer = byteSize.buffer;
        }
        else if (byteSize instanceof Int32Array) {
            this._buffer = byteSize.buffer;
        }
        else if (byteSize instanceof Float32Array) {
            this._buffer = byteSize.buffer;
        }
        else {
            throw new Error(`Invalid constructor argument for ArrayBufferStorage. Expected number of bytes, ArrayBuffer, or a TypedArray. Got ${byteSize} (${(byteSize as any).constructor.name})`);
        }
    }
    get buffer(): ArrayBuffer | null {
        return this._buffer;
    }
}

export class GPUBufferStorage extends UntypedStorage {
    private _buffer: ArrayBuffer | null = null;
    constructor(byteSize: number, alignment: number) {
        super();
    }
    get buffer(): ArrayBuffer | null {
        return this._buffer;
    }
}

export function newTypedArrayForDtype(length: number, dtype: Dtype) {
    switch (dtype) {
        case "uint8":
            return new Uint8Array(length);
        case "int32":
            return new Int32Array(length);
        case "float32":
            return new Float32Array(length);
        default:
            throw new Error(`Unsupported dtype: ${dtype}`);
    }
}

export function newTypedArrayFromArray(
    data: TensorArrayData | null,
    dtype: Dtype,
    allocFor: (shape: Shape) => UntypedStorage
): { storage: UntypedStorage; shape: number[]; strides: number[] } {
    const shape: number[] = [];
    function getShape(data: TensorArrayData | number) {
        if (typeof data === "number") {
            return;
        }
        shape.push(data.length);
        getShape(data[0]);
    }
    if (data !== null) {
        getShape(data);
    }
    const strides = defaultStrides(shape);
    const size = shapeSize(shape);
    const storage = allocFor(shape);
    const flatData = storage.getTypedArray(dtype);
    let flatIndex = 0;
    function flatten(data: TensorArrayData | number) {
        if (typeof data === "number") {
            flatData[flatIndex] = data;
            flatIndex++;
            return;
        }
        for (let i = 0; i < data.length; i++) {
            let d = data[i];
            if (typeof d === "number") {
                for (let j = 0; j < data.length; j++) {
                    flatData[flatIndex] = data[j] as number;
                    flatIndex++;
                }
                return;
            }
            flatten(data[i]);
        }
    }
    if (data !== null) {
        flatten(data);
    }
    return { storage, shape, strides };
}
