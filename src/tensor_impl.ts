import { ATypedArray, Dtype } from "./dtype";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";
import { IDevice } from "./device_if";
import { Deviceish } from "./device";
import { UntypedStorage } from "./storage";
import { ITensor, TensorArrayData } from "./tensor_if";
import { KernelConfigInput, KernelParamsInput } from "./kernel";

export abstract class TensorImpl implements ITensor {
    abstract get storage(): UntypedStorage;
    getTypedArray(): ATypedArray { return this.storage.getTypedArray(this.dtype); }
    abstract get dtype(): Dtype;
    abstract get shape(): Shape;
    abstract get strides(): Strides;
    abstract get device(): IDevice;
    abstract withShape(shape: Shape, strides: Strides): TensorImpl;
    abstract runKernel(name: string, config: KernelConfigInput, params: KernelParamsInput, outputShapes: Shape[], ...additionalInputs: TensorImpl[]): TensorImpl[];

    abstract mm(other: TensorImpl): TensorImpl;
    abstract sum(axis: number | null): TensorImpl;

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

    // Codegen marker
    abstract abs(): TensorImpl;
    abstract abs_(): TensorImpl;
    abstract absGrad(): TensorImpl;
    abstract acos(): TensorImpl;
    abstract acos_(): TensorImpl;
    abstract acosGrad(): TensorImpl;
    abstract acosh(): TensorImpl;
    abstract acosh_(): TensorImpl;
    abstract acoshGrad(): TensorImpl;
    abstract add(other: TensorImpl, alpha?: number): TensorImpl;
    abstract add_(other: TensorImpl, alpha?: number): TensorImpl;
    abstract asin(): TensorImpl;
    abstract asin_(): TensorImpl;
    abstract asinGrad(): TensorImpl;
    abstract asinh(): TensorImpl;
    abstract asinh_(): TensorImpl;
    abstract asinhGrad(): TensorImpl;
    abstract atan(): TensorImpl;
    abstract atan_(): TensorImpl;
    abstract atanGrad(): TensorImpl;
    abstract atan2(other: TensorImpl): TensorImpl;
    abstract atan2_(other: TensorImpl): TensorImpl;
    abstract mul(other: TensorImpl, alpha?: number): TensorImpl;
    abstract mul_(other: TensorImpl, alpha?: number): TensorImpl;
    abstract sub(other: TensorImpl, alpha?: number): TensorImpl;
    abstract sub_(other: TensorImpl, alpha?: number): TensorImpl;
    // End codegen marker
}
