import { Device } from "./device";
import { Dtype } from "./dtype";
import { Shape, Strides, defaultStrides, shapeGetAxis, shapeSize } from "./shape";
import { ITensor, TensorArrayData, TensorImpl } from "./tensor_if";

type ArrayType = Float32Array | Int32Array | Uint8Array;

export class TensorCPU extends TensorImpl {
    private _data: ArrayType;
    private _dtype: Dtype;
    private _shape: number[];
    private _strides: number[];
    private _device: Device;

    get data(): ArrayType {
        return this._data;
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
            return this._data[offset];
        }
        const data = this._data.subarray(offset, offset + this._strides[indices.length]);
        const shape = this._shape.slice(indices.length);
        const strides = this._strides.slice(indices.length);
        return new TensorCPU(data, shape, strides, this._device);
    }

    constructor(data: ArrayType, shape: Shape, strides: Strides, device: Device) {
        super();
        this._data = data;
        this._dtype = "float32";
        if (data instanceof Int32Array) {
            this._dtype = "int32";
        } else if (data instanceof Uint8Array) {
            this._dtype = "boolean";
        }
        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }
    add_(other: ITensor): ITensor {
        if (!(other instanceof TensorCPU)) {
            throw new Error("Only CPU tensors can be added to CPU tensors");
        }
        if (this._shape.length !== other.shape.length) {
            throw new Error(`Shape dimensions must match. Got ${this._shape} and ${other.shape}`);
        }
        for (let i = 0; i < this._shape.length; i++) {
            if (this._shape[i] !== other.shape[i]) {
                throw new Error(`Shapes must match at index ${i}. Got ${this._shape} and ${other.shape}`);
            }
        }
        const od = other.data as ArrayType;
        for (let i = 0; i < this._data.length; i++) {
            this._data[i] += od[i];
        }
        return this;
    }
    mm(other: ITensor): ITensor {
        // Vector dot product?
        if (this._shape.length === 1 && other.shape.length === 1) {
            if (this._shape[0] !== other.shape[0]) {
                throw new Error("Vector dot product requires matching dimensions");
            }
            let sum = 0;
            for (let i = 0; i < this._shape[0]; i++) {
                sum += (this.get(i) as number) * (other.get(i) as number);
            }
            return new TensorCPU(new Float32Array([sum]), [1], [1], this._device);
        }
        // Matrix multiply
        if (this._shape.length !== 2 || other.shape.length !== 2) {
            throw new Error(`Matrix multiply requires 2D tensors. Got ${this._shape.length} and ${other.shape.length}`);
        }
        if (this._shape[1] !== other.shape[0]) {
            throw new Error("Matrix multiply requires matching inner dimensions");
        }
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
            for (let i = 0; i < this._data.length; i++) {
                sum += this._data[i];
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
                    sum += this._data[i * newStrides[0] + j];
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
        let newData = this._data.slice();
        return new TensorCPU(newData, newShape, newStrides, this._device);
    }
}
