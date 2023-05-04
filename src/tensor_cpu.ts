import { Device } from "./device";
import { Dtype } from "./dtype";
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
        shape: Shape,
        strides: Strides,
        device: Device
    ) {
        super();
        this._storage = data;
        this._dtype = "float32";

        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }

    withShape(shape: Shape, strides: Strides): TensorImpl {
        return new TensorCPU(this._storage, shape, strides, this._device);
    }

    add_(other: TensorImpl, alpha?: number): TensorImpl {
        if (!(other instanceof TensorCPU)) {
            throw new Error("Only CPU tensors can be added to CPU tensors");
        }
        const d = this.getTypedArray();
        const od = other.getTypedArray();
        if (alpha === undefined) {
            for (let i = 0; i < d.length; i++) {
                d[i] += od[i];
            }
        }
        else {
            for (let i = 0; i < d.length; i++) {
                d[i] += alpha * od[i];
            }
        }
        return this;
    }
    mm(other: TensorImpl): TensorImpl {
        // Matrix multiply
        const newShape = [this._shape[0], other.shape[1]];
        const [newStorage, newData] = this.device.allocTypedArray(
            newShape[0] * newShape[1],
            this.dtype
        );
        const newStrides = defaultStrides(newShape);
        for (let i = 0; i < newShape[0]; i++) {
            for (let j = 0; j < newShape[1]; j++) {
                let sum = 0;
                for (let k = 0; k < this._shape[1]; k++) {
                    sum +=
                        (this.get(i, k) as number) *
                        (other.get(k, j) as number);
                }
                newData[i * newStrides[0] + j] = sum;
            }
        }
        return new TensorCPU(
            newStorage as ArrayBufferStorage,
            newShape,
            newStrides,
            this._device
        );
    }
    sum(axis: number | null): TensorImpl {
        const d = this.getTypedArray();
        if (axis === null) {
            let sum = 0;
            for (let i = 0; i < d.length; i++) {
                sum += d[i];
            }
            const [newStorage, newData] = this.device.allocTypedArray(
                1,
                this.dtype
            );
            return new TensorCPU(
                newStorage as ArrayBufferStorage,
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
                newShape,
                newStrides,
                this._device
            );
        }
    }
}
