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
        for (let size of []) {
            let bufferPool: { usage: number; buffer: GPUBuffer }[] =
                this._bufferPools[size];
            if (bufferPool === undefined) {
                bufferPool = [];
                this._bufferPools[size] = bufferPool;
            }
            for (let usage of [
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            ]) {
                const count = Math.min(256, (8 * 1024 * 1042) / size);
                for (let i = 0; i < count; i++) {
                    bufferPool.push({
                        usage: usage,
                        buffer: this._device.createBuffer({
                            mappedAtCreation: false,
                            size: size,
                            usage: usage,
                        }),
                    });
                }
            }
        }
    }
    initStorage(shape: Shape, dtype: Dtype, init: (array: ATypedArray) => void): UntypedStorage {
        const elementByteSize = dtypeByteSize(dtype);
        const byteSize = shapeSize(shape) * elementByteSize;
        const buffer = this._device.createBuffer({
            mappedAtCreation: true,
            size: byteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const arrayBuffer = buffer.getMappedRange();
        const array = dtypedBufferToTypedArray(dtype, arrayBuffer);
        init(array);
        buffer.unmap();
        return new GPUBufferStorage(buffer, this.gpuDevice);
    }
    alloc(byteSize: number): UntypedStorage {
        const buffer = this._device.createBuffer({
            mappedAtCreation: false,
            size: byteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        return new GPUBufferStorage(buffer, this.gpuDevice);
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
        return new GPUBufferStorage(buffer, this.gpuDevice);
    }
    getPooledBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
        const sizeRaw = descriptor.size;
        // function nextPowerOfTwo(x: number) {
        //     return Math.pow(2, Math.ceil(Math.log2(x)));
        // }
        // const size = nextPowerOfTwo(sizeRaw);
        const size = sizeRaw;
        let bufferPool: { usage: number; buffer: GPUBuffer }[] =
            this._bufferPools[size];
        if (bufferPool === undefined) {
            bufferPool = [];
            this._bufferPools[size] = bufferPool;
        }
        const npool = bufferPool.length;
        if (npool > 0) {
            // console.log(`Reusing buffer of size ${size} bytes (pool length: ${this._bufferPool.length})`);
            const buffer = bufferPool[npool - 1].buffer;
            bufferPool.splice(npool - 1, 1);
            return buffer;
        } else {
            // console.log(`Creating new buffer of size ${size} bytes (pool length: ${this._bufferPool.length})`);
            // Otherwise, create a new buffer
            return this._device.createBuffer({
                mappedAtCreation: false,
                size: descriptor.size,
                usage: descriptor.usage,
            });
        }
    }
    createKernel(spec: KernelSpec, config: KernelConfig): Kernel {
        return new KernelWebGPU(spec, config, this);
    }
}
