import { Shape, shapeSize } from "./shape";
import { ATypedArray, Dtype, dtypeByteSize } from "./dtype";
import type { UntypedStorage, BufferHeap, HeapBuffer } from "./storage";
import {
    Kernel,
    KernelConfig,
    KernelConfigInput,
    KernelKey,
    KernelSpec,
    getKernelConfig,
    getKernelKey,
} from "./kernel";
import { registry as kernelRegistry } from "./kernels";

export type DeviceType = "cpu" | "webgpu";
export type DeviceId = string;

export type Deviceish = DeviceType | Device | DeviceId;

export abstract class Device {
    private _id: DeviceId;
    private _type: DeviceType;
    private _kernels: { [key: KernelKey]: Kernel } = {};
    get id(): DeviceId {
        return this._id;
    }
    get type(): DeviceType {
        return this._type;
    }
    abstract get workgroupMaxSize(): [number, number, number];
    abstract get workgroupMaxCount(): number;
    constructor(id: DeviceId, type: DeviceType) {
        this._id = id;
        this._type = type;
    }
    abstract alloc(byteSize: number): UntypedStorage;
    allocFor(shape: Shape, dtype: Dtype): UntypedStorage {
        const elementByteSize = dtypeByteSize(dtype);
        const byteSize = shapeSize(shape) * elementByteSize;
        return this.alloc(byteSize);
    }
    heaps: BufferHeap<GPUBuffer | ArrayBuffer>[] = [];
    abstract allocBufferHeap(): BufferHeap<GPUBuffer | ArrayBuffer>;
    heapAlloc(byteSize: number): HeapBuffer<GPUBuffer | ArrayBuffer> {
        for (let heap of this.heaps) {
            const buffer = heap.alloc(byteSize);
            if (buffer !== null) {
                return buffer;
            }
        }
        const heap = this.allocBufferHeap();
        this.heaps.push(heap);
        const result = heap.alloc(byteSize);
        if (result === null) {
            throw new Error(`Out of memory when trying to allocate buffer of size ${byteSize}. Heap size is ${heap.size}.`);
        }
        return result;
    }
    getKernel(name: string, config: KernelConfigInput): Kernel {
        const spec = kernelRegistry[name];
        if (spec === undefined) {
            throw new Error(`Kernel "${name}" not found`);
        }
        const kconfig = getKernelConfig(spec, config);
        const key = getKernelKey(spec, kconfig);
        let kernel = this._kernels[key];
        if (kernel === undefined) {
            kernel = this.createKernel(spec, kconfig);
            this._kernels[key] = kernel;
        }
        return kernel;
    }
    abstract createKernel(spec: KernelSpec, config: KernelConfig): Kernel;
    abstract getBufferForKernel(
        storage: UntypedStorage,
        dtype: Dtype
    ): ATypedArray | GPUBuffer;
    abstract getStorageFromKernel(
        buffer: ATypedArray | GPUBuffer,
        pooled: boolean
    ): UntypedStorage;
}
