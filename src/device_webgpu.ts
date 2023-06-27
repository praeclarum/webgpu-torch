import { Device } from "./device";
import { ATypedArray, Dtype, dtypeByteSize, dtypedBufferToTypedArray } from "./dtype";
import { GPUBufferStorage, UntypedStorage, BufferHeap, HeapBuffer } from "./storage";
import type { Kernel, KernelConfig, KernelSpec } from "./kernel";
import { KernelWebGPU } from "./kernel_webgpu";
import { Shape, shapeSize } from "./shape";

export class DeviceWebGPU extends Device {
    private _device: GPUDevice;
    private _bufferPools: {
        [size: number]: { usage: number; buffer: GPUBuffer }[];
    } = {};
    private _finalizationRegistry: FinalizationRegistry<GPUBuffer>;
    get gpuDevice(): GPUDevice {
        return this._device;
    }
    get workgroupMaxSize(): [number, number, number] {
        return [
            this._device.limits.maxComputeWorkgroupSizeX,
            this._device.limits.maxComputeWorkgroupSizeY,
            this._device.limits.maxComputeWorkgroupSizeZ,
        ];
    }
    get workgroupMaxCount(): number {
        return this._device.limits.maxComputeWorkgroupsPerDimension;
    }
    constructor(id: string, adapter: GPUAdapter, device: GPUDevice) {
        super(id, "webgpu");
        this._device = device;
        this._finalizationRegistry = new FinalizationRegistry<GPUBuffer>(
            (buffer) => {
                const size = buffer.size;
                let bufferPool: { usage: number; buffer: GPUBuffer }[] =
                    this._bufferPools[size];
                if (bufferPool === undefined) {
                    bufferPool = [];
                    this._bufferPools[size] = bufferPool;
                }
                if (bufferPool.length < 256) {
                    bufferPool.push({
                        usage: buffer.usage,
                        buffer: buffer,
                    });
                    // console.log("Added buffer to pool", buffer);
                }
            }
        );
    }
    initStorage(shape: Shape, dtype: Dtype, init: (array: ATypedArray) => void): UntypedStorage {
        const elementByteSize = dtypeByteSize(dtype);
        const byteSize = shapeSize(shape) * elementByteSize;
        const alignedByteSize = Math.floor((byteSize + 3) / 4) * 4;
        const buffer = this._device.createBuffer({
            mappedAtCreation: true,
            size: alignedByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const arrayBuffer = buffer.getMappedRange();
        const array = dtypedBufferToTypedArray(dtype, arrayBuffer);
        init(array);
        buffer.unmap();
        return new GPUBufferStorage(buffer, this);
    }
    alloc(byteSize: number): UntypedStorage {
        const alignedByteSize = Math.floor((byteSize + 3) / 4) * 4;
        const buffer = this._device.createBuffer({
            mappedAtCreation: false,
            size: alignedByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        return new GPUBufferStorage(buffer, this);
    }
    allocBufferHeap(): BufferHeap<GPUBuffer> {
        const byteSize = this.gpuDevice.limits.maxBufferSize;
        const buffer = this._device.createBuffer({
            mappedAtCreation: false,
            size: byteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const minOrder = Math.ceil(Math.log2(this.gpuDevice.limits.minStorageBufferOffsetAlignment));
        return new BufferHeap<GPUBuffer>(buffer, byteSize, minOrder);
    }
    createHeapStorage(buffer: HeapBuffer<GPUBuffer>): UntypedStorage {
        return new GPUBufferStorage(buffer, this);
    }
    createKernel(spec: KernelSpec, config: KernelConfig): Kernel {
        return new KernelWebGPU(spec, config, this);
    }
}
