import { ATypedArray, Dtype } from "./dtype";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";
import { IDevice } from "./device_if";
import { Deviceish } from "./device";
import { UntypedStorage } from "./storage";

export type TensorArrayData = Array<number | TensorArrayData>;

export interface ITensor {
    get storage(): UntypedStorage;
    get dtype(): Dtype;
    get shape(): Shape;
    get device(): IDevice;
    get(...indices: number[]): number | ITensor;
    add_(other: ITensor): ITensor;
    expand(shape: Shape): ITensor;
    mm(other: ITensor): ITensor;
    sum(axis: number | null): ITensor;
    t(): ITensor;
}

export abstract class TensorImpl implements ITensor {
    abstract get storage(): UntypedStorage;
    getTypedArray(): ATypedArray { return this.storage.getTypedArray(this.dtype); }
    abstract get dtype(): Dtype;
    abstract get shape(): Shape;
    abstract get strides(): Strides;
    abstract get device(): IDevice;
    abstract withShape(shape: Shape, strides: Strides): TensorImpl;

    abstract add_(other: TensorImpl): TensorImpl;
    abstract mm(other: TensorImpl): TensorImpl;
    abstract sum(axis: number | null): TensorImpl;

    /** Returns a new view of this tensor with singleton dimensions expanded to a larger size.
    Passing -1 as the size for a dimension means not changing the size of that dimension.
    Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front. For the new dimensions, the size cannot be set to -1.
    Expanding a tensor does not allocate new memory, but only creates a new view on the existing tensor where a dimension of size one is expanded to a larger size by setting the stride to 0. Any dimension of size 1 can be expanded to an arbitrary value without allocating new memory. */
    expand(shape: Shape): TensorImpl {
        const newShape = shape.slice();
        const newStrides = Array(newShape.length).fill(0);
        // Update newStrides based on the current strides
        // so that the expansion happens
        // in the correct direction
        let j = newShape.length - 1;
        let thisShape = this.shape;
        let thisStrides = this.strides;
        for (let i = thisShape.length - 1; i >= 0; i--) {
            if (thisShape[i] === 1) {
                newStrides[j] = 0;
            } else {
                newStrides[j] = thisStrides[i];
                j--;
            }
            if (newShape[j] === -1) {
                newShape[j] = thisShape[i];
            }
        }
        return this.withShape(newShape, newStrides);
    }

    get(...indices: number[]): number | TensorImpl {
        let thisStrides = this.strides;
        let array = this.getTypedArray();
        let offset = 0;
        for (let i = 0; i < indices.length; i++) {
            offset += indices[i] * thisStrides[i];
        }
        if (thisStrides.length === indices.length) {
            return array[offset];
        }
        throw new Error("Cannot get value from tensor with wrong number of indices");
    }

    t(): TensorImpl {
        let newShape = this.shape.slice();
        newShape.reverse();
        let newStrides = this.strides.slice();
        newStrides.reverse();
        return this.withShape(newShape, newStrides);
    }
}

export type TensorJsonData = {
    data: TensorArrayData | TensorImpl;
    dtype: Dtype;
    requiresGrad?: boolean;
    device?: Deviceish;
};
