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

    withShape(shape: Shape, strides: Strides): TensorImpl {
        return new TensorWebGPU(
            this._storage,
            this._dtype,
            shape,
            strides,
            this._device
        );
    }

    add_(other: TensorImpl): TensorImpl {
        const result = this.device.zeros(this.shape, this.dtype);
        throw new Error("Add_ not implemented.");
        return result;
    }
    mm(other: TensorImpl): TensorImpl {
        const resultRows = this.shape[0];
        const resultCols = other.shape[1];
        const result = this.device.zeros([resultRows, resultCols], this.dtype);
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

        throw new Error("MM not implemented.");
        return result;
    }
    sum(axis: number | null): TensorImpl {
        throw new Error("Sum not implemented.");
    }
}
