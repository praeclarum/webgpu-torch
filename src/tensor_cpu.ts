import { Device } from "./device";
import { ATypedArray, Dtype } from "./dtype";
import { KernelConfigInput, KernelParamsInput } from "./kernel";
import {
    Shape,
    Strides,
    defaultStrides,
    shapeGetAxis,
    shapeSize,
} from "./shape";
import { ArrayBufferStorage } from "./storage";
import { TensorImpl } from "./tensor_impl";

export class TensorCPU extends TensorImpl {
    private _storage: ArrayBufferStorage;
    private _dtype: Dtype;
    private _shape: number[];
    private _strides: number[];
    private _device: Device;

    get storage(): ArrayBufferStorage {
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
        data: ArrayBufferStorage,
        dtype: Dtype,
        shape: Shape,
        strides: Strides,
        device: Device
    ) {
        super();
        this._storage = data;
        this._dtype = dtype;
        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }
    withShape(shape: Shape, strides: Strides): TensorImpl {
        return new TensorCPU(
            this._storage,
            this._dtype,
            shape,
            strides,
            this._device
        );
    }

    runKernelInplace(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        ...additionalInputs: TensorCPU[]
    ): TensorCPU {
        const kernel = this._device.getKernel(name, config);
        const inputBuffers = additionalInputs.map((t) => t.getTypedArray());
        const outputBuffers = [this.getTypedArray()];
        kernel.run(inputBuffers, params, outputBuffers);
        return this;
    }
    runKernel(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        outputShapes: Shape[],
        ...additionalInputs: TensorCPU[]
    ): TensorCPU[] {
        const kernel = this.device.getKernel(name, config);
        const inputBuffers = [
            this.getTypedArray(),
            ...additionalInputs.map((t) => t.getTypedArray()),
        ];
        const outputBuffers = kernel.run(inputBuffers, params) as ATypedArray[];
        if (outputBuffers.length !== outputShapes.length) {
            throw new Error(
                `Expected ${outputShapes.length} output buffers (given the provided outputShapes to runKernel), but got ${outputBuffers.length} output buffers when running the kernel "${name}".`
            );
        }
        return outputBuffers.map(
            (outputBuffer, i) =>
                new TensorCPU(
                    new ArrayBufferStorage(outputBuffer.buffer),
                    this.dtype,
                    outputShapes[i],
                    defaultStrides(outputShapes[i]),
                    this._device
                )
        );
    }

    sum(axis: number | null): TensorImpl {
        const d = this.getTypedArray();
        if (axis === null) {
            if (true) {
                const config = {
                    dtype: this.dtype,
                    workgroupSize: 64,
                };
                const params = {
                    size: shapeSize(this.shape),
                };
                return this.runKernel("sum", config, params, [[]])[0];
            }
            let sum = 0;
            for (let i = 0; i < d.length; i++) {
                sum += d[i];
            }
            const [newStorage, newData] = this.device.allocTypedArray(
                1,
                this.dtype
            );
            newData[0] = sum;
            return new TensorCPU(
                newStorage as ArrayBufferStorage,
                this._dtype,
                [],
                [],
                this._device
            );
        } else {
            axis = shapeGetAxis(this._shape, axis);
            const newShape = this._shape.slice();
            newShape.splice(axis, 1);
            const [newStorage, newData] = this.device.allocTypedArray(
                shapeSize(newShape),
                this.dtype
            );
            const newStrides = defaultStrides(newShape);
            const axisStride = this._strides[axis];
            for (let i = 0; i < newData.length; i++) {
                let sum = 0;
                for (let j = 0; j < axisStride; j++) {
                    sum += d[i * newStrides[0] + j];
                }
                newData[i] = sum;
            }
            return new TensorCPU(
                newStorage as ArrayBufferStorage,
                this._dtype,
                newShape,
                newStrides,
                this._device
            );
        }
    }
}
