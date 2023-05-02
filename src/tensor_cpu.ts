import { Device } from "./device";
import { Dtype, ATypedArray } from "./dtype";
import { Shape, Strides, defaultStrides, shapeGetAxis, shapeSize } from "./shape";
import { ITensor, TensorArrayData, TensorImpl } from "./tensor_if";

export class TensorCPU extends TensorImpl {
    private _typedStorage: ATypedArray;
    private _dtype: Dtype;
    private _shape: number[];
    private _strides: number[];
    private _device: Device;

    get typedStorage(): ATypedArray {
        return this._typedStorage;
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

    get(...indices: number[]): number | ITensor {
        let offset = 0;
        for (let i = 0; i < indices.length; i++) {
            offset += indices[i] * this._strides[i];
        }
        if (this._strides.length === indices.length) {
            return this._typedStorage[offset];
        }
        const data = this._typedStorage.subarray(offset, offset + this._strides[indices.length]);
        const shape = this._shape.slice(indices.length);
        const strides = this._strides.slice(indices.length);
        return new TensorCPU(data, shape, strides, this._device);
    }

    constructor(data: ATypedArray, shape: Shape, strides: Strides, device: Device) {
        super();
        this._typedStorage = data;
        this._dtype = "float32";
        
        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }
    add_(other: ITensor): ITensor {
        if (!(other instanceof TensorCPU)) {
            throw new Error("Only CPU tensors can be added to CPU tensors");
        }
        const od = other.typedStorage as ATypedArray;
        for (let i = 0; i < this._typedStorage.length; i++) {
            this._typedStorage[i] += od[i];
        }
        return this;
    }
    /** Returns a new view of this tensor with singleton dimensions expanded to a larger size.
    Passing -1 as the size for a dimension means not changing the size of that dimension.
    Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front. For the new dimensions, the size cannot be set to -1.
    Expanding a tensor does not allocate new memory, but only creates a new view on the existing tensor where a dimension of size one is expanded to a larger size by setting the stride to 0. Any dimension of size 1 can be expanded to an arbitrary value without allocating new memory. */
    expand(shape: Shape): ITensor {
        const newShape = shape.slice();
        const newStrides = Array(newShape.length).fill(0);
        // Update newStrides based on the current strides
        // so that the expansion happens
        // in the correct direction
        let j = newShape.length - 1;
        for (let i = this._shape.length - 1; i >= 0; i--) {
            if (this._shape[i] === 1) {
                newStrides[j] = 0;
            } else {
                newStrides[j] = this._strides[i];
                j--;
            }
            if (newShape[j] === -1) {
                newShape[j] = this._shape[i];
            }
        }
        return new TensorCPU(this._typedStorage, newShape, newStrides, this._device);
    }
    mm(other: ITensor): ITensor {
        // Matrix multiply
        const newShape = [this._shape[0], other.shape[1]];
        const newData = new Float32Array(newShape[0] * newShape[1]);
        const newStrides = defaultStrides(newShape);
        for (let i = 0; i < newShape[0]; i++) {
            for (let j = 0; j < newShape[1]; j++) {
                let sum = 0;
                for (let k = 0; k < this._shape[1]; k++) {
                    sum += (this.get(i, k) as number) * (other.get(k, j) as number);
                }
                newData[i * newStrides[0] + j] = sum;
            }
        }
        return new TensorCPU(newData, newShape, newStrides, this._device);
    }
    sum(axis: number | null): ITensor {
        if (axis === null) {
            let sum = 0;
            for (let i = 0; i < this._typedStorage.length; i++) {
                sum += this._typedStorage[i];
            }
            return new TensorCPU(new Float32Array([sum]), [], [], this._device);
        }
        else {
            axis = shapeGetAxis(this._shape, axis);
            const newShape = this._shape.slice();
            newShape.splice(axis, 1);
            const newData = new Float32Array(shapeSize(newShape));
            const newStrides = defaultStrides(newShape);
            const axisStride = this._strides[axis];
            for (let i = 0; i < newData.length; i++) {
                let sum = 0;
                for (let j = 0; j < axisStride; j++) {
                    sum += this._typedStorage[i * newStrides[0] + j];
                }
                newData[i] = sum;
            }
            return new TensorCPU(newData, newShape, newStrides, this._device);
        }
    }
    t(): ITensor {
        let newShape = this._shape.slice();
        newShape.reverse();
        let newStrides = this._strides.slice();
        newStrides.reverse();
        return new TensorCPU(this._typedStorage, newShape, newStrides, this._device);
    }
}
