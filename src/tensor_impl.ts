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

    abstract runKernelInplace(name: string, config: KernelConfigInput, params: KernelParamsInput, ...additionalInputs: TensorImpl[]): TensorImpl;
    abstract runKernel(name: string, config: KernelConfigInput, params: KernelParamsInput, outputShapes: Shape[], ...additionalInputs: TensorImpl[]): TensorImpl[];
    
    abstract mm(other: TensorImpl): TensorImpl;
    abstract sum(axis: number | null): TensorImpl;

    get isContiguous(): boolean {
        let strides = this.strides;
        let shape = this.shape;
        let offset = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            if (strides[i] !== offset) {
                return false;
            }
            offset *= shape[i];
        }
        return true;
    }
    get isScalar(): boolean {
        return this.shape.length === 0 || (this.shape.length === 1 && this.shape[0] === 1);
    }

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
        // console.log("EXPAND", this.shape, this.strides, shape, newShape, newStrides);
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
    abs(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("abs", { dtype: this.dtype }, params, [this.shape])[0];
    }
    abs_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("abs_", { dtype: this.dtype }, params);
    }
    acos(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("acos", { dtype: this.dtype }, params, [this.shape])[0];
    }
    acos_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acos_", { dtype: this.dtype }, params);
    }
    acosh(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("acosh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    acosh_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acosh_", { dtype: this.dtype }, params);
    }
    add(other: TensorImpl, alpha?: number): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernel("add", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    add_(other: TensorImpl, alpha?: number): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("add_", { dtype: this.dtype }, params, other);
    }
    asin(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("asin", { dtype: this.dtype }, params, [this.shape])[0];
    }
    asin_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asin_", { dtype: this.dtype }, params);
    }
    asinh(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("asinh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    asinh_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asinh_", { dtype: this.dtype }, params);
    }
    atan(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("atan", { dtype: this.dtype }, params, [this.shape])[0];
    }
    atan_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("atan_", { dtype: this.dtype }, params);
    }
    atan2(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("atan2", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    atan2_(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("atan2_", { dtype: this.dtype }, params, other);
    }
    ceil(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("ceil", { dtype: this.dtype }, params, [this.shape])[0];
    }
    ceil_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("ceil_", { dtype: this.dtype }, params);
    }
    copysign(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("copysign", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    copysign_(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("copysign_", { dtype: this.dtype }, params, other);
    }
    cos(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("cos", { dtype: this.dtype }, params, [this.shape])[0];
    }
    cos_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cos_", { dtype: this.dtype }, params);
    }
    cosh(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("cosh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    cosh_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cosh_", { dtype: this.dtype }, params);
    }
    deg2rad(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("deg2rad", { dtype: this.dtype }, params, [this.shape])[0];
    }
    deg2rad_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("deg2rad_", { dtype: this.dtype }, params);
    }
    div(other: TensorImpl, alpha?: number): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernel("div", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    div_(other: TensorImpl, alpha?: number): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("div_", { dtype: this.dtype }, params, other);
    }
    exp(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("exp", { dtype: this.dtype }, params, [this.shape])[0];
    }
    exp_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp_", { dtype: this.dtype }, params);
    }
    exp2(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("exp2", { dtype: this.dtype }, params, [this.shape])[0];
    }
    exp2_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp2_", { dtype: this.dtype }, params);
    }
    expm1(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("expm1", { dtype: this.dtype }, params, [this.shape])[0];
    }
    expm1_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("expm1_", { dtype: this.dtype }, params);
    }
    floor(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("floor", { dtype: this.dtype }, params, [this.shape])[0];
    }
    floor_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("floor_", { dtype: this.dtype }, params);
    }
    floor_divide(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("floor_divide", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    floor_divide_(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("floor_divide_", { dtype: this.dtype }, params, other);
    }
    frac(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("frac", { dtype: this.dtype }, params, [this.shape])[0];
    }
    frac_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("frac_", { dtype: this.dtype }, params);
    }
    hypot(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("hypot", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    hypot_(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("hypot_", { dtype: this.dtype }, params, other);
    }
    ldexp(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("ldexp", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    ldexp_(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("ldexp_", { dtype: this.dtype }, params, other);
    }
    log(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("log", { dtype: this.dtype }, params, [this.shape])[0];
    }
    log_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log_", { dtype: this.dtype }, params);
    }
    log10(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("log10", { dtype: this.dtype }, params, [this.shape])[0];
    }
    log10_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log10_", { dtype: this.dtype }, params);
    }
    log1p(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("log1p", { dtype: this.dtype }, params, [this.shape])[0];
    }
    log1p_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log1p_", { dtype: this.dtype }, params);
    }
    log2(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("log2", { dtype: this.dtype }, params, [this.shape])[0];
    }
    log2_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log2_", { dtype: this.dtype }, params);
    }
    logaddexp(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("logaddexp", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    logaddexp_(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("logaddexp_", { dtype: this.dtype }, params, other);
    }
    logaddexp2(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("logaddexp2", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    logaddexp2_(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("logaddexp2_", { dtype: this.dtype }, params, other);
    }
    mul(other: TensorImpl, alpha?: number): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernel("mul", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    mul_(other: TensorImpl, alpha?: number): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("mul_", { dtype: this.dtype }, params, other);
    }
    neg(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("neg", { dtype: this.dtype }, params, [this.shape])[0];
    }
    neg_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("neg_", { dtype: this.dtype }, params);
    }
    positive(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("positive", { dtype: this.dtype }, params, [this.shape])[0];
    }
    positive_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("positive_", { dtype: this.dtype }, params);
    }
    pow(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("pow", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    pow_(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("pow_", { dtype: this.dtype }, params, other);
    }
    rad2deg(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("rad2deg", { dtype: this.dtype }, params, [this.shape])[0];
    }
    rad2deg_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rad2deg_", { dtype: this.dtype }, params);
    }
    reciprocal(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("reciprocal", { dtype: this.dtype }, params, [this.shape])[0];
    }
    reciprocal_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("reciprocal_", { dtype: this.dtype }, params);
    }
    round(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("round", { dtype: this.dtype }, params, [this.shape])[0];
    }
    round_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("round_", { dtype: this.dtype }, params);
    }
    rsqrt(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("rsqrt", { dtype: this.dtype }, params, [this.shape])[0];
    }
    rsqrt_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rsqrt_", { dtype: this.dtype }, params);
    }
    sigmoid(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sigmoid", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sigmoid_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sigmoid_", { dtype: this.dtype }, params);
    }
    sign(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sign", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sign_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sign_", { dtype: this.dtype }, params);
    }
    sin(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sin", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sin_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sin_", { dtype: this.dtype }, params);
    }
    sinc(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sinc", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sinc_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinc_", { dtype: this.dtype }, params);
    }
    sinh(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sinh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sinh_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinh_", { dtype: this.dtype }, params);
    }
    sqrt(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sqrt", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sqrt_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sqrt_", { dtype: this.dtype }, params);
    }
    square(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("square", { dtype: this.dtype }, params, [this.shape])[0];
    }
    square_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("square_", { dtype: this.dtype }, params);
    }
    sub(other: TensorImpl, alpha?: number): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernel("sub", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    sub_(other: TensorImpl, alpha?: number): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("sub_", { dtype: this.dtype }, params, other);
    }
    tan(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("tan", { dtype: this.dtype }, params, [this.shape])[0];
    }
    tan_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tan_", { dtype: this.dtype }, params);
    }
    tanh(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("tanh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    tanh_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tanh_", { dtype: this.dtype }, params);
    }
    trunc(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("trunc", { dtype: this.dtype }, params, [this.shape])[0];
    }
    trunc_(): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("trunc_", { dtype: this.dtype }, params);
    }
    xlogy(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("xlogy", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    xlogy_(other: TensorImpl): TensorImpl {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("xlogy_", { dtype: this.dtype }, params, other);
    }
    // End codegen marker
}
