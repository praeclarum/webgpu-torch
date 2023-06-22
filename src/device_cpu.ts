import { Device } from "./device";
import { ArrayBufferStorage, BufferHeap, HeapBuffer, UntypedStorage } from "./storage";
import { KernelCPU } from "./kernel_cpu";
import { dtypeByteSize, type ATypedArray, type Dtype } from "./dtype";
import type { Kernel, KernelConfig, KernelSpec } from "./kernel";
import { Shape, shapeSize } from "./shape";

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
    initStorage(shape: Shape, dtype: Dtype, init: (array: ATypedArray) => void): ArrayBufferStorage {
        const elementByteSize = dtypeByteSize(dtype);
        const byteSize = shapeSize(shape) * elementByteSize;
        const storage = new ArrayBufferStorage(byteSize);
        init(storage.getTypedArray(dtype));
        return storage;
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
    createHeapStorage(buffer: HeapBuffer<ArrayBuffer>): UntypedStorage {
        return new ArrayBufferStorage(buffer);
    }
    createKernel(spec: KernelSpec, config: KernelConfig): Kernel {
        return new KernelCPU(spec, config, this);
    }
}
