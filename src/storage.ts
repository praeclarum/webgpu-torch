import { Device } from "./device";
import { ATypedArray, Dtype, dtypeArrayCtors } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";

export type TensorArrayData = Array<number | TensorArrayData>;

export abstract class UntypedStorage {
    abstract get byteSize(): number;
    abstract get mappedArrayBuffer(): ArrayBuffer | null;
    abstract get isMapped(): boolean;
    abstract mapReadAsync(): Promise<void>;
    abstract unmap(): void;
    abstract destroy(): void;
    getTypedArray(dtype: Dtype): ATypedArray {
        const buffer = this.mappedArrayBuffer;
        if (buffer === null) {
            throw new Error("Storage is not mapped");
        }
        if (dtype in dtypeArrayCtors) {
            return new dtypeArrayCtors[dtype](buffer);
        }
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
}

export class ArrayBufferStorage extends UntypedStorage {
    private _buffer: ArrayBuffer;
    get cpuBuffer(): ArrayBuffer {
        return this._buffer;
    }
    get byteSize(): number {
        return this._buffer.byteLength;
    }
    get mappedArrayBuffer(): ArrayBuffer | null {
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
    async mapReadAsync(): Promise<void> {
        // Do nothing
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
    private _mappedArrayBuffer: [GPUBuffer, ArrayBuffer|null] | null = null;
    private _device: GPUDevice;
    get byteSize(): number {
        return this._buffer.size;
    }
    get gpuBuffer(): GPUBuffer {
        return this._buffer;
    }
    get gpuDevice(): GPUDevice {
        return this._device;
    }
    get mappedArrayBuffer(): ArrayBuffer | null {
        if (this._mappedArrayBuffer !== null) {
            let ar = this._mappedArrayBuffer[1];
            if (ar === null) {
                ar = this._mappedArrayBuffer[0].getMappedRange();
                this._mappedArrayBuffer[1] = ar;
            }
            return ar;
        }
        return null;
    }
    constructor(buffer: GPUBuffer, device: GPUDevice)
    constructor(byteSize: number, device: GPUDevice, usage: GPUBufferUsageFlags, )
    constructor(input: number|GPUBuffer, device: GPUDevice, usage?: GPUBufferUsageFlags) {
        super();
        this._device = device;
        if (input instanceof GPUBuffer) {
            this._buffer = input;
            switch (this._buffer.mapState) {
                case "mapped":
                    this._mappedArrayBuffer = [this._buffer, null];
                    break;
                case "unmapped":
                    this._mappedArrayBuffer = null;
                    break;
                case "pending":
                    throw new Error("GPUBuffer is pending. Please wait for it to finish mapping before creating a GPUBufferStorage with it.");
            }
        }
        else if (typeof input === "number" && usage !== undefined && device !== undefined) {
            this._buffer = device.createBuffer({
                mappedAtCreation: true,
                size: input,
                usage: usage,
            });
            this._mappedArrayBuffer = [this._buffer, null];
        }
        else {
            throw new Error(`Invalid constructor arguments for GPUBufferStorage. Expected GPUBuffer, or byteSize, usage, and device. Got ${input} (${(input as any).constructor.name})`);
        }
    }
    get isMapped(): boolean {
        return this._buffer.mapState === "mapped";
    }
    async mapReadAsync(): Promise<void> {
        if (this.isMapped) {
            return;
        }
        let mapBuffer: GPUBuffer = this._buffer;
        if ((this._buffer.usage & GPUBufferUsage.MAP_READ) === 0) {
            mapBuffer = this.copyBufferToReadableBuffer();
        }
        await mapBuffer.mapAsync(GPUMapMode.READ);
        if (mapBuffer.mapState === "mapped") {
            this._mappedArrayBuffer = [mapBuffer, null];
        }
        else {
            throw new Error("GPUBuffer failed to map");
        }
    }
    unmap(): void {
        this._mappedArrayBuffer = null;
        this._buffer.unmap();
    }
    destroy(): void {
        this._mappedArrayBuffer = null;
        this._buffer.destroy();
    }
    private copyBufferToReadableBuffer(): GPUBuffer {
        const size = this._buffer.size;
        const readBuffer = this._device.createBuffer({
            mappedAtCreation: false,
            size: size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        // console.log("copyBufferToReadableBuffer", this._buffer, readBuffer);
        // Encode commands for copying outputs to readable buffers
        const commandEncoder = this._device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this._buffer /* source buffer */,
            0 /* source offset */,
            readBuffer /* destination buffer */,
            0 /* destination offset */,
            size /* size */
        );
        // Submit GPU commands
        const gpuCommands = commandEncoder.finish();
        this._device.queue.submit([gpuCommands]);
        return readBuffer;
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
    device: Device): { storage: UntypedStorage; shape: number[]; strides: number[] } {
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
    const storage = device.allocFor(shape, dtype);
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
