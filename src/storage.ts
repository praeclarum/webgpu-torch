import { ATypedArray, Dtype, dtypeArrayCtors } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { TensorArrayData } from "./tensor_if";

export abstract class UntypedStorage {
    abstract get byteSize(): number;
    abstract get buffer(): ArrayBuffer | null;
    abstract get isMapped(): boolean;
    abstract mapReadAsync(): Promise<boolean>;
    abstract unmap(): void;
    abstract destroy(): void;
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
    get byteSize(): number {
        return this._buffer.byteLength;
    }
    get buffer(): ArrayBuffer | null {
        return this._buffer;
    }
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
    get isMapped(): boolean {
        return true;
    }
    async mapReadAsync(): Promise<boolean> {
        return true;
    }
    unmap(): void {
        // Do nothing
    }
    destroy(): void {
        // Do nothing
    }
}

export class GPUBufferStorage extends UntypedStorage {
    private _buffer: GPUBuffer;
    private _mappedArrayBuffer: ArrayBuffer | null = null;
    get byteSize(): number {
        return this._buffer.size;
    }
    get gpuBuffer(): GPUBuffer {
        return this._buffer;
    }
    get buffer(): ArrayBuffer | null {
        if (this._mappedArrayBuffer !== null) {
            return this._mappedArrayBuffer;
        }
        if (this._buffer.mapState === "mapped") {
            this._mappedArrayBuffer = this._buffer.getMappedRange();
            return this._mappedArrayBuffer;
        }
        return null;
    }
    constructor(buffer: GPUBuffer)
    constructor(byteSize: number, usage: GPUBufferUsageFlags, device: GPUDevice)
    constructor(input: number|GPUBuffer, usage?: GPUBufferUsageFlags, device?: GPUDevice) {
        super();
        if (input instanceof GPUBuffer) {
            this._buffer = input;
        }
        else if (typeof input === "number" && usage !== undefined && device !== undefined) {
            this._buffer = device.createBuffer({
                mappedAtCreation: true,
                size: input,
                usage: usage,
            });
        }
        else {
            throw new Error(`Invalid constructor arguments for GPUBufferStorage. Expected GPUBuffer, or byteSize, usage, and device. Got ${input} (${(input as any).constructor.name})`);
        }
    }
    get isMapped(): boolean {
        return this._buffer.mapState === "mapped";
    }
    async mapReadAsync(): Promise<boolean> {
        if (this.isMapped) {
            return true;
        }
        await this._buffer.mapAsync(GPUMapMode.READ);
        return this.isMapped;
    }
    unmap(): void {
        this._mappedArrayBuffer = null;
        this._buffer.unmap();
    }
    destroy(): void {
        this._mappedArrayBuffer = null;
        this._buffer.destroy();
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
    function flatten(data: TensorArrayData) {
        for (let i = 0; i < data.length; i++) {
            let d = data[i];
            if (typeof d === "number") {
                for (let j = 0; j < data.length; j++) {
                    flatData[flatIndex] = data[j] as number;
                    flatIndex++;
                }
                return;
            }
            if (d instanceof Array) {
                flatten(d);
            }
            else {
                throw new Error(`Invalid data type: ${d} (${(d as any).constructor.name})`);
            }
        }
    }
    if (data !== null) {
        flatten(data);
    }
    return { storage, shape, strides };
}
