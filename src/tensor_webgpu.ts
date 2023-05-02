import { Device } from "./device";
import { DeviceWebGPU } from "./device_webgpu";
import { Dtype } from "./dtype";
import { Shape, Strides } from "./shape";
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
        const result = this._device.zeros(this.shape, this.dtype);
        // TODO: Implement add_ for GPU tensors
        return result;
    }
    mm(other: TensorWebGPU): TensorWebGPU {
        const resultRows = this.shape[0];
        const resultCols = other.shape[1];
        const result = this._device.zeros([resultRows, resultCols], this.dtype);
        const resultMatrixBufferSize = result.storage.byteSize;
        const device = this._device.device;
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
                        buffer: result.gpuBuffer,
                    },
                },
            ],
        });

        const shaderModule = device.createShaderModule({
            code: `
              struct Matrix {
                size : vec2f,
                numbers: array<f32>,
              }
          
              @group(0) @binding(0) var<storage, read> firstMatrix : Matrix;
              @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
              @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;
          
              @compute @workgroup_size(8, 8)
              fn main(@builtin(global_invocation_id) global_id : vec3u) {
                // Guard against out-of-bounds work group sizes
                if (global_id.x >= u32(firstMatrix.size.x) || global_id.y >= u32(secondMatrix.size.y)) {
                  return;
                }
          
                resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);
          
                let resultCell = vec2(global_id.x, global_id.y);
                var result = 0.0;
                for (var i = 0u; i < u32(firstMatrix.size.y); i = i + 1u) {
                  let a = i + resultCell.x * u32(firstMatrix.size.y);
                  let b = resultCell.y + i * u32(secondMatrix.size.y);
                  result = result + firstMatrix.numbers[a] * secondMatrix.numbers[b];
                }
          
                let index = resultCell.y + resultCell.x * u32(secondMatrix.size.y);
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
        const gpuReadBuffer = device.createBuffer({
            size: resultMatrixBufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        // Encode commands for copying buffer to buffer.
        commandEncoder.copyBufferToBuffer(
            result.gpuBuffer /* source buffer */,
            0 /* source offset */,
            gpuReadBuffer /* destination buffer */,
            0 /* destination offset */,
            resultMatrixBufferSize /* size */
        );
        
        // Submit GPU commands.
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        // Read buffer.
        const readStorage = new GPUBufferStorage(gpuReadBuffer);
        const readTensor = new TensorWebGPU(
            readStorage,
            this.dtype,
            result.shape,
            result.strides,
            this._device);
        return readTensor;
    }
    sum(axis: number | null): TensorImpl {
        throw new Error("Sum not implemented.");
    }
}
