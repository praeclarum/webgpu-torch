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
    abstract acos(): TensorImpl;
    abstract acos_(): TensorImpl;
    abstract acosh(): TensorImpl;
    abstract acosh_(): TensorImpl;
    abstract add(other: TensorImpl, alpha?: number): TensorImpl;
    abstract add_(other: TensorImpl, alpha?: number): TensorImpl;
    abstract asin(): TensorImpl;
    abstract asin_(): TensorImpl;
    abstract asinh(): TensorImpl;
    abstract asinh_(): TensorImpl;
    abstract atan(): TensorImpl;
    abstract atan_(): TensorImpl;
    abstract atan2(other: TensorImpl): TensorImpl;
    abstract atan2_(other: TensorImpl): TensorImpl;
    abstract ceil(): TensorImpl;
    abstract ceil_(): TensorImpl;
    abstract copysign(other: TensorImpl): TensorImpl;
    abstract copysign_(other: TensorImpl): TensorImpl;
    abstract cos(): TensorImpl;
    abstract cos_(): TensorImpl;
    abstract cosh(): TensorImpl;
    abstract cosh_(): TensorImpl;
    abstract deg2rad(): TensorImpl;
    abstract deg2rad_(): TensorImpl;
    abstract div(other: TensorImpl, alpha?: number): TensorImpl;
    abstract div_(other: TensorImpl, alpha?: number): TensorImpl;
    abstract exp(): TensorImpl;
    abstract exp_(): TensorImpl;
    abstract exp2(): TensorImpl;
    abstract exp2_(): TensorImpl;
    abstract expm1(): TensorImpl;
    abstract expm1_(): TensorImpl;
    abstract floor(): TensorImpl;
    abstract floor_(): TensorImpl;
    abstract floor_divide(other: TensorImpl): TensorImpl;
    abstract floor_divide_(other: TensorImpl): TensorImpl;
    abstract frac(): TensorImpl;
    abstract frac_(): TensorImpl;
    abstract hypot(other: TensorImpl): TensorImpl;
    abstract hypot_(other: TensorImpl): TensorImpl;
    abstract ldexp(other: TensorImpl): TensorImpl;
    abstract ldexp_(other: TensorImpl): TensorImpl;
    abstract log(): TensorImpl;
    abstract log_(): TensorImpl;
    abstract log10(): TensorImpl;
    abstract log10_(): TensorImpl;
    abstract log1p(): TensorImpl;
    abstract log1p_(): TensorImpl;
    abstract log2(): TensorImpl;
    abstract log2_(): TensorImpl;
    abstract logaddexp(other: TensorImpl): TensorImpl;
    abstract logaddexp_(other: TensorImpl): TensorImpl;
    abstract logaddexp2(other: TensorImpl): TensorImpl;
    abstract logaddexp2_(other: TensorImpl): TensorImpl;
    abstract mul(other: TensorImpl, alpha?: number): TensorImpl;
    abstract mul_(other: TensorImpl, alpha?: number): TensorImpl;
    abstract neg(): TensorImpl;
    abstract neg_(): TensorImpl;
    abstract positive(): TensorImpl;
    abstract positive_(): TensorImpl;
    abstract pow(other: TensorImpl): TensorImpl;
    abstract pow_(other: TensorImpl): TensorImpl;
    abstract rad2deg(): TensorImpl;
    abstract rad2deg_(): TensorImpl;
    abstract reciprocal(): TensorImpl;
    abstract reciprocal_(): TensorImpl;
    abstract round(): TensorImpl;
    abstract round_(): TensorImpl;
    abstract rsqrt(): TensorImpl;
    abstract rsqrt_(): TensorImpl;
    abstract sigmoid(): TensorImpl;
    abstract sigmoid_(): TensorImpl;
    abstract sign(): TensorImpl;
    abstract sign_(): TensorImpl;
    abstract sin(): TensorImpl;
    abstract sin_(): TensorImpl;
    abstract sinc(): TensorImpl;
    abstract sinc_(): TensorImpl;
    abstract sinh(): TensorImpl;
    abstract sinh_(): TensorImpl;
    abstract sqrt(): TensorImpl;
    abstract sqrt_(): TensorImpl;
    abstract square(): TensorImpl;
    abstract square_(): TensorImpl;
    abstract sub(other: TensorImpl, alpha?: number): TensorImpl;
    abstract sub_(other: TensorImpl, alpha?: number): TensorImpl;
    abstract tan(): TensorImpl;
    abstract tan_(): TensorImpl;
    abstract tanh(): TensorImpl;
    abstract tanh_(): TensorImpl;
    abstract trunc(): TensorImpl;
    abstract trunc_(): TensorImpl;
    abstract xlogy(other: TensorImpl): TensorImpl;
    abstract xlogy_(other: TensorImpl): TensorImpl;
    // End codegen marker
}
