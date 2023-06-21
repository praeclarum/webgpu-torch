import { Device } from "./device";
import { ArrayBufferStorage, BufferHeap, HeapBuffer, UntypedStorage } from "./storage";
import { KernelCPU } from "./kernel_cpu";
import type { ATypedArray, Dtype } from "./dtype";
import type { Kernel, KernelConfig, KernelSpec } from "./kernel";

export class DeviceCPU extends Device {
    get workgroupMaxSize(): [number, number, number] {
        return [
            256,
            256,
            64];
    }
    get workgroupMaxCount(): number {
        return 65535;
    }
    constructor() {
        super("cpu", "cpu");
    }
    alloc(byteSize: number): UntypedStorage {
        return new ArrayBufferStorage(byteSize);
    }
    allocBufferHeap(): BufferHeap<GPUBuffer | ArrayBuffer> {
        const size = 256*1024*1024;
        const array = new ArrayBuffer(size);
        const minOrder = 8; // Align to 256 bytes
        return new BufferHeap<ArrayBuffer>(array, size, minOrder);
    }
    createStorage(buffer: HeapBuffer<ArrayBuffer>): UntypedStorage {
        return new ArrayBufferStorage(buffer);
    }
    createKernel(spec: KernelSpec, config: KernelConfig): Kernel {
        return new KernelCPU(spec, config, this);
    }
    getStorageFromKernel(storage: ATypedArray | GPUBuffer, pooled: boolean): UntypedStorage {
        if (
            storage instanceof Uint8Array ||
            storage instanceof Uint32Array ||
            storage instanceof Int32Array ||
            storage instanceof Float32Array
        ) {
            return new ArrayBufferStorage(storage.buffer);
        }
        throw new Error(
            `Cannot wrap buffer of type ${storage.constructor.name} to get CPU storage`
        );
    }
    getBufferForKernel(
        storage: UntypedStorage,
        dtype: Dtype
    ): ATypedArray | GPUBuffer {
        if (storage instanceof ArrayBufferStorage) {
            return storage.getTypedArray(dtype);
        }
        throw new Error(
            `Cannot unwrap buffer of type ${storage.constructor.name} to get CPU buffer`
        );
    }
}
