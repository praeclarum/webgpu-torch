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
    private _heapEnabled = false;
    private _heaps: BufferHeap<GPUBuffer | ArrayBuffer>[] = [];
    private _heapFinalizers: FinalizationRegistry<HeapBuffer<GPUBuffer | ArrayBuffer>>;
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
        this._heapFinalizers = new FinalizationRegistry<HeapBuffer<GPUBuffer | ArrayBuffer>>(
            (buffer) => {
                // console.log("Finalizing heap buffer", buffer);
                buffer.free();
            }
        );
    }
    abstract initStorage(shape: Shape, dtype: Dtype, init: (array: ATypedArray) => void): UntypedStorage;
    abstract allocBufferHeap(): BufferHeap<GPUBuffer | ArrayBuffer>;
    abstract createHeapStorage(buffer: HeapBuffer<GPUBuffer | ArrayBuffer>): UntypedStorage;
    abstract alloc(byteSize: number): UntypedStorage;
    heapAlloc(byteSize: number): UntypedStorage {
        if (this._heapEnabled === false) {
            return this.alloc(byteSize);
        }
        let resultBuffer: HeapBuffer<GPUBuffer | ArrayBuffer> | null = null;
        for (let heap of this._heaps) {
            const buffer = heap.alloc(byteSize);
            if (buffer !== null) {
                resultBuffer = buffer;
            }
        }
        if (resultBuffer === null) {
            let totalMB = 0;
            for (let heap of this._heaps) {
                totalMB += heap.size / 1024 / 1024;
            }
            if (totalMB > 8*1024) {
                // console.log(`Total heap size is ${totalMB.toFixed(2)} MB. Not allocating more.`);
                return this.alloc(byteSize);
            }
            const heap = this.allocBufferHeap();
            this._heaps.push(heap);
            // console.log(`Allocated buffer heap of size ${heap.size} (${totalMB.toFixed(2)} MB)`);
            resultBuffer = heap.alloc(byteSize);
            if (resultBuffer === null) {
                throw new Error(`Out of memory when trying to allocate buffer of size ${byteSize}. Heap size is ${heap.size}.`);
            }
        }
        const storage = this.createHeapStorage(resultBuffer);
        this._heapFinalizers.register(storage, resultBuffer);
        // console.log("Allocated heap buffer of size", resultBuffer.byteSize);
        return storage;
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
}
