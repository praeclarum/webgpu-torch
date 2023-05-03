import { Device } from "./device";
import { DeviceWebGPU } from "./device_webgpu";
import { Dtype, dtypeByteSize } from "./dtype";
import { Shape, Strides, defaultStrides } from "./shape";
import { GPUBufferStorage } from "./storage";
import { TensorImpl } from "./tensor_if";

export class TensorWebGPU extends TensorImpl {
    private _storage: GPUBufferStorage;
    private _dtype: Dtype;
    private _shape: number[];
    private _strides: number[];
    private _device: DeviceWebGPU;

    get gpuBuffer(): GPUBuffer {
        return this._storage.gpuBuffer;
    }

    get storage(): GPUBufferStorage {
        return this._storage;
    }
    get dtype(): Dtype {
        return this._dtype;
    }
    get shape(): number[] {
        return this._shape;
    }
    get strides(): number[] {
        return this._strides;
    }
    get device(): Device {
        return this._device;
    }

    constructor(
        storage: GPUBufferStorage,
        dtype: Dtype,
        shape: Shape,
        strides: Strides,
        device: DeviceWebGPU
    ) {
        super();
        this._storage = storage;
        this._dtype = dtype;
        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }

    withShape(shape: Shape, strides: Strides): TensorWebGPU {
        return new TensorWebGPU(
            this._storage,
            this._dtype,
            shape,
            strides,
            this._device
        );
    }

    add_(other: TensorWebGPU): TensorWebGPU {
        const kernel = this._device.getKernel("Add", { resultDtype: "f32" });
        const params = {
            resultSize: this.shape.reduce((a, b) => a * b),
        };
        this.gpuBuffer.unmap();
        other.gpuBuffer.unmap();
        const outputs = kernel.run([this.gpuBuffer, other.gpuBuffer], params);
        const readBuffer = outputs[0];
        const readStorage = new GPUBufferStorage(readBuffer);
        const resultShape = this.shape;
        const readTensor = new TensorWebGPU(
            readStorage,
            this.dtype,
            resultShape,
            defaultStrides(resultShape),
            this._device
        );
        return readTensor;
    }
    mm(other: TensorWebGPU): TensorWebGPU {
        const kernel = this._device.getKernel("MM", { resultDtype: "f32" });
        const params = {
            resultRows: this.shape[0],
            resultCols: other.shape[1],
            innerDim: this.shape[1],
            alpha: 1.0,
        };
        this.gpuBuffer.unmap();
        other.gpuBuffer.unmap();
        const outputs = kernel.run([this.gpuBuffer, other.gpuBuffer], params);
        const readBuffer = outputs[0];
        const readStorage = new GPUBufferStorage(readBuffer);
        const resultShape = [params.resultRows, params.resultCols];
        const readTensor = new TensorWebGPU(
            readStorage,
            this.dtype,
            resultShape,
            defaultStrides(resultShape),
            this._device
        );
        return readTensor;
    }
    mm_old(other: TensorWebGPU): TensorWebGPU {
        const resultRows = this.shape[0];
        const resultCols = other.shape[1];
        const elementByteSize = dtypeByteSize(this.dtype);
        const resultBufferSize = resultRows * resultCols * elementByteSize;
        const device = this._device.device;
        const paramsBufferSize = 16;
        const paramsBuffer = device.createBuffer({
            mappedAtCreation: true,
            size: paramsBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const paramsArrayBuffer = paramsBuffer.getMappedRange();
        const paramsInt32Array = new Int32Array(paramsArrayBuffer);
        paramsInt32Array[0] = this.shape[0];
        paramsInt32Array[1] = other.shape[1];
        paramsInt32Array[2] = this.shape[1];
        const paramsFloat32Array = new Float32Array(paramsArrayBuffer);
        paramsFloat32Array[3] = 1.0;
        const resultBuffer = device.createBuffer({
            mappedAtCreation: false,
            size: resultBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        paramsBuffer.unmap();
        this.gpuBuffer.unmap();
        other.gpuBuffer.unmap();
        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage" as GPUBufferBindingType,
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage" as GPUBufferBindingType,
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage" as GPUBufferBindingType,
                    },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage" as GPUBufferBindingType,
                    },
                },
            ],
        });
        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.gpuBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: other.gpuBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: resultBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: paramsBuffer,
                    },
                },
            ],
        });

        const shaderModule = device.createShaderModule({
            code: `
              struct Matrix {
                numbers: array<f32>,
              }

              struct MMParameters {
                resultRows : u32,
                resultCols : u32,
                innerDim : u32,
                alpha : f32,
              }
          
              @group(0) @binding(0) var<storage, read> firstMatrix : Matrix;
              @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
              @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;
              @group(0) @binding(3) var<storage, read> parameters : MMParameters;
          
              @compute @workgroup_size(8, 8)
              fn main(@builtin(global_invocation_id) global_id : vec3u) {
                if (global_id.x >= parameters.resultRows || global_id.y >= u32(parameters.resultCols)) {
                  return;
                }
                var result = 0.0;
                for (var i = 0u; i < parameters.innerDim; i = i + 1u) {
                  let a = global_id.x * parameters.innerDim + i;
                  let b = i * parameters.resultCols + global_id.y;
                  result = result + firstMatrix.numbers[a] * secondMatrix.numbers[b];
                }
                let index = global_id.y + global_id.x * parameters.resultCols;
                resultMatrix.numbers[index] = result;
              }
            `,
        });

        const computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });

        const commandEncoder = device.createCommandEncoder();

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil(resultRows / 8);
        const workgroupCountY = Math.ceil(resultCols / 8);
        passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
        passEncoder.end();

        // Get a GPU buffer for reading in an unmapped state.
        const readBuffer = device.createBuffer({
            mappedAtCreation: false,
            size: resultBufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Encode commands for copying buffer to buffer.
        commandEncoder.copyBufferToBuffer(
            resultBuffer /* source buffer */,
            0 /* source offset */,
            readBuffer /* destination buffer */,
            0 /* destination offset */,
            resultBufferSize /* size */
        );

        // Submit GPU commands.
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        // Read buffer.
        const readStorage = new GPUBufferStorage(readBuffer);
        const readTensor = new TensorWebGPU(
            readStorage,
            this.dtype,
            [resultRows, resultCols],
            defaultStrides([resultRows, resultCols]),
            this._device
        );
        return readTensor;
    }
    sum(axis: number | null): TensorImpl {
        throw new Error("Sum not implemented.");
    }
}
