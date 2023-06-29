import { Device } from "./device";
import { DeviceWebGPU } from "./device_webgpu";
import { cpuDevice } from "./devices";
import { ATypedArray, Dtype, dtypedBufferToTypedArray } from "./dtype";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";

export type TensorArrayData = Array<number | TensorArrayData>;

export abstract class UntypedStorage {
    abstract get device(): Device;
    abstract get byteOffset(): number;
    abstract get byteSize(): number;
    abstract destroy(): void;
    abstract clone(): UntypedStorage;
    abstract toTypedArrayAsync(dtype: Dtype): Promise<ATypedArray>;
    async toArrayAsync(dtype: Dtype): Promise<number[]> {
        const typedArray = await this.toTypedArrayAsync(dtype);
        return Array.from(typedArray);
    }
}

export class ArrayBufferStorage extends UntypedStorage {
    private readonly _buffer: ArrayBuffer;
    private readonly _byteOffset: number = 0;
    private readonly _byteSize: number;
    private readonly _heapBuffer: HeapBuffer<ArrayBuffer> | null = null;
    get device(): Device {
        return cpuDevice;
    }
    get cpuBuffer(): ArrayBuffer {
        return this._buffer;
    }
    get byteOffset(): number {
        return this._byteOffset;
    }
    get byteSize(): number {
        return this._byteSize;
    }
    constructor(byteSize: number | ArrayBuffer | ATypedArray | HeapBuffer<ArrayBuffer>) {
        super();
        if (typeof byteSize === "number") {
            this._buffer = new ArrayBuffer(byteSize as number);
            this._byteSize = byteSize;
        } else if (byteSize instanceof HeapBuffer) {
            this._buffer = byteSize.heap.buffer;
            this._byteOffset = byteSize.offset;
            this._byteSize = byteSize.byteSize;
            this._heapBuffer = byteSize;
        } else if (byteSize instanceof ArrayBuffer) {
            this._buffer = byteSize;
            this._byteSize = byteSize.byteLength;
        } else if (byteSize instanceof Uint8Array) {
            this._buffer = byteSize.buffer;
            this._byteSize = byteSize.byteLength;
        } else if (byteSize instanceof Int32Array) {
            this._buffer = byteSize.buffer;
            this._byteSize = byteSize.byteLength;
        } else if (byteSize instanceof Float32Array) {
            this._buffer = byteSize.buffer;
            this._byteSize = byteSize.byteLength;
        } else {
            throw new Error(
                `Invalid constructor argument for ArrayBufferStorage. Expected number of bytes, ArrayBuffer, or a TypedArray. Got ${byteSize} (${
                    (byteSize as any).constructor.name
                })`
            );
        }
    }
    getTypedArray(dtype: Dtype): ATypedArray {
        return dtypedBufferToTypedArray(dtype, this._buffer, this._byteOffset, this._byteSize);
    }
    async toTypedArrayAsync(dtype: Dtype): Promise<ATypedArray> {
        return this.getTypedArray(dtype);
    }
    destroy(): void {
        // Do nothing
    }
    clone(): UntypedStorage {
        return new ArrayBufferStorage(this._buffer.slice(0));
    }
}

export class GPUBufferStorage extends UntypedStorage {
    private readonly _device: DeviceWebGPU;
    private readonly _byteOffset: number = 0;
    private readonly _byteSize: number;
    private readonly _buffer: GPUBuffer;
    private _gpuDevice: GPUDevice;
    get device(): Device {
        return this._device;
    }
    get byteSize(): number {
        return this._byteSize;
    }
    get byteOffset(): number {
        return this._byteOffset;
    }
    get gpuBuffer(): GPUBuffer {
        return this._buffer;
    }
    get gpuDevice(): GPUDevice {
        return this._gpuDevice;
    }
    constructor(buffer: GPUBuffer, device: DeviceWebGPU);
    constructor(buffer: HeapBuffer<GPUBuffer>, device: DeviceWebGPU);
    constructor(
        byteSize: number,
        device: DeviceWebGPU,
        usage: GPUBufferUsageFlags
    );
    constructor(
        input: number | GPUBuffer | HeapBuffer<GPUBuffer>,
        device: DeviceWebGPU,
        usage?: GPUBufferUsageFlags
    ) {
        super();
        this._device = device;
        this._gpuDevice = device.gpuDevice;
        if (input instanceof GPUBuffer) {
            this._buffer = input;
            this._byteSize = this._buffer.size;
        } else if (input instanceof HeapBuffer) {
            this._buffer = input.heap.buffer;
            this._byteOffset = input.offset;
            this._byteSize = input.byteSize;
            // console.log(`Created GPUBufferStorage from HeapBuffer with offset ${input.offset}, size ${input.byteSize}`);
        } else if (
            typeof input === "number" &&
            usage !== undefined &&
            device !== undefined
        ) {
            const alignedByteSize = Math.floor((input + 3) / 4) * 4;
            this._buffer = this._gpuDevice.createBuffer({
                mappedAtCreation: true,
                size: alignedByteSize,
                usage: usage,
            });
            this._byteSize = this._buffer.size;
        } else {
            throw new Error(
                `Invalid constructor arguments for GPUBufferStorage. Expected GPUBuffer, or byteSize, usage, and device. Got ${input} (${
                    (input as any).constructor.name
                })`
            );
        }
    }
    destroy(): void {
        this._buffer.destroy();
    }
    async toTypedArrayAsync(dtype: Dtype): Promise<ATypedArray> {
        const mapBuffer = this.copyBufferToReadableBuffer();
        await mapBuffer.mapAsync(GPUMapMode.READ);
        if (mapBuffer.mapState !== "mapped") {
            throw new Error("GPUBuffer failed to map");
        }
        const arrayBuffer = mapBuffer.getMappedRange();
        const array = dtypedBufferToTypedArray(dtype, arrayBuffer);
        return array;
    }
    private copyBufferToReadableBuffer(): GPUBuffer {
        const size = this._byteSize;
        const readBuffer = this._gpuDevice.createBuffer({
            mappedAtCreation: false,
            size: size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        // console.log("copyBufferToReadableBuffer", this._buffer, readBuffer);
        // Encode commands for copying outputs to readable buffers
        const commandEncoder = this._gpuDevice.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this._buffer /* source buffer */,
            this._byteOffset /* source offset */,
            readBuffer /* destination buffer */,
            0 /* destination offset */,
            size /* size */
        );
        // Submit GPU commands
        const gpuCommands = commandEncoder.finish();
        this._gpuDevice.queue.submit([gpuCommands]);
        return readBuffer;
    }
    clone(): UntypedStorage {
        const usage =
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.STORAGE;
        const cloneBuffer = this._gpuDevice.createBuffer({
            mappedAtCreation: false,
            size: this._buffer.size,
            usage: usage,
        });
        const commandEncoder = this._gpuDevice.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this._buffer /* source buffer */,
            0 /* source offset */,
            cloneBuffer /* destination buffer */,
            0 /* destination offset */,
            this._buffer.size /* size */
        );
        // Submit GPU commands
        const gpuCommands = commandEncoder.finish();
        this._gpuDevice.queue.submit([gpuCommands]);
        return new GPUBufferStorage(cloneBuffer, this._device);
    }
}

export function newStorageFromATypedArray(
    data: ATypedArray,
    shape: Shape,
    dtype: Dtype,
    device: Device
): { storage: UntypedStorage; shape: Shape; strides: Strides } {
    const strides = defaultStrides(shape);
    const storage = device.initStorage(shape, dtype, (flatData) => {
        const elementSize = flatData.BYTES_PER_ELEMENT;
        if (elementSize > 1 && data instanceof Uint8Array) {
            const bflatData = new Uint8Array(flatData.buffer);
            if (bflatData.length !== data.length) {
                throw new Error(`TypedArray from Uint8Array length mismatch: ${bflatData.length} !== ${data.length}`);
            }
            bflatData.set(data);
        }
        else if (flatData.length !== data.length) {
            throw new Error(`TypedArray length mismatch: ${flatData.length} !== ${data.length}`);
        }
        else {
            flatData.set(data);
        }
    });
    return {
        storage,
        shape,
        strides,
    }
}

export function newTypedArrayFromArray(
    data: TensorArrayData,
    dtype: Dtype,
    device: Device
): { storage: UntypedStorage; shape: Shape; strides: Strides } {
    const shape: number[] = [];
    function getShape(data?: TensorArrayData | number) {
        if (data === undefined || typeof data === "number") {
            return;
        }
        shape.push(data.length);
        getShape(data[0]);
    }
    if (data !== null) {
        getShape(data);
    }
    const strides = defaultStrides(shape);
    const storage = device.initStorage(shape, dtype, (flatData) => {
        // const flatData = storage.getTypedArray(dtype);
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
                } else {
                    throw new Error(
                        `Invalid data type: ${d} (${(d as any).constructor.name})`
                    );
                }
            }
        }
        if (data !== null) {
            flatten(data);
        }
    });
    return { storage, shape, strides };
}

export function flatDataToArray(data: ATypedArray, shape: Shape, strides: Strides): TensorArrayData | number {
    const rank = shape.length;
    if (rank == 0) {
        return data[0];
    }
    if (rank == 1 && shape[0] == 1) {
        return [data[0]];
    }
    if (rank == 1 && strides[0] == 1) {
        return Array.from(data);
    }

    const index: number[] = [];
    return readArray(index);

    function readArray(index: number[]): TensorArrayData {
        const dim = index.length;

        if (dim == shape.length - 1) {
            const offset = calculateOffset(index);
            const length = shape[dim];
            const subarray = data.subarray(offset, offset + length);
            if (subarray.length !== length) {
                throw new Error(`Failed to get sub array for index [${index}] (tensor shape [${shape}] and strides [${strides}]) at offset ${offset} with length ${length} from buffer of length ${data.length}`);
            }
            return Array.from(subarray);
        } else {
            const result: TensorArrayData = [];
            for (let i = 0; i < shape[dim]; i++) {
                index.push(i);
                result.push(readArray(index));
                index.pop();
            }
            return result;
        }
    }

    function calculateOffset(index: number[]): number {
        let offset = 0;
        for (let i = 0; i < index.length; i++) {
            offset += index[i] * strides[i];
        }
        return offset;
    }
}

function getNearestPowerOfTwo(size: GPUSize64): GPUSize64 {
    return Math.pow(2, Math.ceil(Math.log2(size)));
}

export class HeapBuffer<T> {
    readonly heap: BufferHeap<T>;
    readonly offset: number;
    readonly order: number;
    get byteSize(): number {
        return 1 << this.order;
    }
    constructor(heap: BufferHeap<T>, offset: number, order: number) {
        this.heap = heap;
        this.offset = offset;
        this.order = order;
    }
    free(): void {
        this.heap.free(this);
    }
}

export class BufferHeap<T> {
    private readonly _heapSize: number;
    private readonly _heapBuffer: T;
    private readonly _minOrder: number;
    private orderFreeLists: number[][];

    get buffer(): T {
        return this._heapBuffer;
    }
    get size(): number {
        return this._heapSize;
    }

    constructor(heapBuffer: T, heapSize: number, minOrder: number) {
        this._heapSize = getNearestPowerOfTwo(heapSize);
        if (this._heapSize > heapSize) {
            throw new Error(
                `Requested heap size ${heapSize} is not a power of two`
            );
        }
        this._heapBuffer = heapBuffer;
        this._minOrder = minOrder;
        this.orderFreeLists = [];
        const maxOrder = Math.log2(this._heapSize);
        for (let i = 0; i <= maxOrder; ++i) {
            this.orderFreeLists.push([]);
        }
        this.orderFreeLists[maxOrder].push(0);
    }

    private static getBuddyOffset(offset: number, order: number): number {
        return offset ^ (1 << order);
    }

    alloc(size: number): HeapBuffer<T> | null {
        const order = Math.max(
            this._minOrder,
            Math.log2(getNearestPowerOfTwo(size))
        );
        // console.log(
        //     `Allocating ${1 << order} bytes (order ${order}) for request of ${size} bytes`
        // );
        let allocOrder = order;
        while (
            allocOrder < this.orderFreeLists.length &&
            this.orderFreeLists[allocOrder].length === 0
        ) {
            allocOrder++;
        }
        if (allocOrder === this.orderFreeLists.length) {
            // console.error(`No available memory block found.`);
            return null;
        }
        const offset = this.orderFreeLists[allocOrder].pop()!;
        const result = new HeapBuffer(this, offset, order);
        while (allocOrder > order) {
            allocOrder--;
            const buddyOffset = BufferHeap.getBuddyOffset(offset, allocOrder);
            this.orderFreeLists[allocOrder].push(buddyOffset);
        }
        return result;
    }

    free(buffer: HeapBuffer<T>): void {
        let offset = buffer.offset;
        let order = buffer.order;
        while (order < this.orderFreeLists.length - 1) {
            const buddyOffset = BufferHeap.getBuddyOffset(offset, order);
            const index = this.orderFreeLists[order].indexOf(buddyOffset);
            if (index === -1) {
                break;
            }
            this.orderFreeLists[order].splice(index, 1);
            order++;
            offset = Math.min(offset, buddyOffset);
        }
        this.orderFreeLists[order].push(offset);
    }
}
