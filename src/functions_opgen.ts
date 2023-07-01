import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import type { Tensor } from "./tensor";
import { shapeSize, defaultStrides, broadcastShapes, stridedShapeIsContiguous } from "./shape";
export class AbsFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("abs", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("abs_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class AcosFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acos", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acos_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class AcoshFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acosh", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acosh_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class AddFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number | undefined];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
                alpha: alpha || 1.0,
            };
            return input.runKernel("add_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                    alpha: alpha || 1.0,
                };
                return input.runKernel("add_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                    alpha: alpha || 1.0,
                };
                return input.runKernel("add", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number | undefined];
        ctx.alpha = alpha;
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
                alpha: ctx.alpha || 1.0,
            };
            return input.runKernel("add_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
                alpha: ctx.alpha || 1.0,
            };
            return input.runKernel("add_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class AsinFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asin", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asin_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class AsinhFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asinh", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asinh_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class AtanFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atan", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atan_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class Atan2Function extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("atan2_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("atan2_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("atan2", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("atan2_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("atan2_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class CeilFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("ceil", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("ceil_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class CopysignFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("copysign_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("copysign_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("copysign", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("copysign_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("copysign_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class CosFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("cos", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("cos_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class CoshFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("cosh", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("cosh_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class Deg2radFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("deg2rad", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("deg2rad_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class DivFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("div_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("div_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("div", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("div_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("div_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class ExpFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("exp", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("exp_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class Exp2Function extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("exp2", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("exp2_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class Expm1Function extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("expm1", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("expm1_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class FloorFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("floor", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("floor_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class FracFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("frac", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("frac_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class HypotFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("hypot_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("hypot_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("hypot", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("hypot_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("hypot_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class LdexpFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("ldexp_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("ldexp_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("ldexp", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("ldexp_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("ldexp_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class LogFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("log", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("log_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class Log10Function extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("log10", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("log10_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class Log1pFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("log1p", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("log1p_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class Log2Function extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("log2", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("log2_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class LogaddexpFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("logaddexp_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("logaddexp_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("logaddexp", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("logaddexp_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("logaddexp_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class Logaddexp2Function extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("logaddexp2_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("logaddexp2_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("logaddexp2", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("logaddexp2_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("logaddexp2_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class MulFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("mul_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("mul_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("mul", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("mul_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("mul_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class NegFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("neg", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("neg_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class PositiveFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("positive", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("positive_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class PowFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("pow_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("pow_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("pow", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("pow_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("pow_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class Rad2degFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("rad2deg", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("rad2deg_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class ReciprocalFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("reciprocal", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("reciprocal_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class ReluFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("relu", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("relu_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class RoundFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("round", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("round_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class RsqrtFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("rsqrt", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("rsqrt_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class SigmoidFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sigmoid", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sigmoid_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class SignFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sign", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sign_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class SiluFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("silu", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("silu_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class SinFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sin", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sin_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class SincFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sinc", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sinc_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class SinhFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sinh", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sinh_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class SqrtFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sqrt", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("sqrt_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class SquareFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("square", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("square_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class SubFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number | undefined];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
                alpha: alpha || 1.0,
            };
            return input.runKernel("sub_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                    alpha: alpha || 1.0,
                };
                return input.runKernel("sub_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                    alpha: alpha || 1.0,
                };
                return input.runKernel("sub", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number | undefined];
        ctx.alpha = alpha;
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
                alpha: ctx.alpha || 1.0,
            };
            return input.runKernel("sub_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
                alpha: ctx.alpha || 1.0,
            };
            return input.runKernel("sub_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class TanFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("tan", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("tan_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class TanhFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("tanh", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("tanh_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class TruncFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("trunc", {"dtype":"float32"}, params, [input.shape])[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("trunc_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
    }
}
export class XlogyFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("xlogy_scalar", {"dtype":"float32"}, params, [input.shape])[0];
        } else {
            const broadcasted = broadcastShapes(input, other);
            if (!stridedShapeIsContiguous(broadcasted.a) || !stridedShapeIsContiguous(broadcasted.b)) {
                const inputDims = broadcasted.a.shape.length;
                const otherDims = broadcasted.b.shape.length;
                if (inputDims > 4 || otherDims > 4) {
                    throw new Error("Broadcasting not supported for tensors with more than 4 dimensions");
                }
                const params = {
                    inputStrides0: inputDims > 0 ? broadcasted.a.strides[0] : 0,
                    otherStrides0: otherDims > 0 ? broadcasted.b.strides[0] : 0,
                    outputStrides0: broadcasted.output.shape.length > 0 ? broadcasted.output.strides[0] : 1,
                    inputStrides1: inputDims > 1 ? broadcasted.a.strides[1] : 0,
                    otherStrides1: otherDims > 1 ? broadcasted.b.strides[1] : 0,
                    outputStrides1: broadcasted.output.shape.length > 1 ? broadcasted.output.strides[1] : 1,
                    inputStrides2: inputDims > 2 ? broadcasted.a.strides[2] : 0,
                    otherStrides2: otherDims > 2 ? broadcasted.b.strides[2] : 0,
                    outputStrides2: broadcasted.output.shape.length > 2 ? broadcasted.output.strides[2] : 1,
                    inputStrides3: inputDims > 3 ? broadcasted.a.strides[3] : 0,
                    otherStrides3: otherDims > 3 ? broadcasted.b.strides[3] : 0,
                    outputStrides3: broadcasted.output.shape.length > 3 ? broadcasted.output.strides[3] : 1,
                    size: shapeSize(broadcasted.output.shape),
                };
                return input.runKernel("xlogy_strided", {"dtype":"float32"}, params, [broadcasted.output.shape], other)[0];
            } else {
                if (shapeSize(input.shape) !== shapeSize(other.shape)) {
                    throw new Error(`Shape sizes must match. Got ${input.shape} and ${other.shape}`);
                }
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("xlogy", {"dtype":"float32"}, params, [input.shape], other)[0];
            }
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        if (typeof other === "number") {
            const params = {
                size: shapeSize(input.shape),
                other: other,
            };
            return input.runKernel("xlogy_scalar_grad", {"dtype":"float32"}, params, [input.shape], outputGrad);
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("xlogy_grad", {"dtype":"float32"}, params, [input.shape, other.shape], other, outputGrad);
        }
    }
}
export class AllFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        if (dim !== undefined) {
            dim = Array.isArray(dim) && dim.length === 1 ? dim[0] : dim;
            if (typeof dim === "number") {
                const inputShape = input.shape;
                let outputShape = input.shape.slice();
                outputShape[dim] = 1;
                let outputStrides = defaultStrides(outputShape);
                const params = {
                    size: shapeSize(outputShape),
                    inputShape0: input.shape.length > 0 ? input.shape[0] : 1,
                    inputStride0: input.shape.length > 0 ? input.strides[0] : 1,
                    outputStride0: outputShape.length > 0 ? outputStrides[0] : 1,
                    inputShape1: input.shape.length > 1 ? input.shape[1] : 1,
                    inputStride1: input.shape.length > 1 ? input.strides[1] : 1,
                    outputStride1: outputShape.length > 1 ? outputStrides[1] : 1,
                    inputShape2: input.shape.length > 2 ? input.shape[2] : 1,
                    inputStride2: input.shape.length > 2 ? input.strides[2] : 1,
                    outputStride2: outputShape.length > 2 ? outputStrides[2] : 1,
                    inputShape3: input.shape.length > 3 ? input.shape[3] : 1,
                    inputStride3: input.shape.length > 3 ? input.strides[3] : 1,
                    outputStride3: outputShape.length > 3 ? outputStrides[3] : 1,
                };
                if (!keepdim) outputShape.splice(dim, 1);
                return input.runKernel("all_dim", {dim,maxdim:inputShape.length,dtype:"float32"}, params, [outputShape])[0];
            } else {
                throw new Error("Multi-dimension reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("all", {"dtype":"float32","workgroupSize":256}, params, [[]])[0];
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        ctx.dim = dim;
        ctx.keepdim = keepdim;
        ctx.saveForBackward(input, output);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, output] = ctx.savedTensors as [Tensor, Tensor];
        const dim: number | number[] | undefined = ctx.dim;
        const keepdim: boolean | undefined = ctx.keepdim;
        if (dim !== undefined) {
            if (typeof dim === "number") {
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("all_dim_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
            } else {
                throw new Error("Multi-dimension backward reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("all_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
        }
    }
}
export class AnyFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        if (dim !== undefined) {
            dim = Array.isArray(dim) && dim.length === 1 ? dim[0] : dim;
            if (typeof dim === "number") {
                const inputShape = input.shape;
                let outputShape = input.shape.slice();
                outputShape[dim] = 1;
                let outputStrides = defaultStrides(outputShape);
                const params = {
                    size: shapeSize(outputShape),
                    inputShape0: input.shape.length > 0 ? input.shape[0] : 1,
                    inputStride0: input.shape.length > 0 ? input.strides[0] : 1,
                    outputStride0: outputShape.length > 0 ? outputStrides[0] : 1,
                    inputShape1: input.shape.length > 1 ? input.shape[1] : 1,
                    inputStride1: input.shape.length > 1 ? input.strides[1] : 1,
                    outputStride1: outputShape.length > 1 ? outputStrides[1] : 1,
                    inputShape2: input.shape.length > 2 ? input.shape[2] : 1,
                    inputStride2: input.shape.length > 2 ? input.strides[2] : 1,
                    outputStride2: outputShape.length > 2 ? outputStrides[2] : 1,
                    inputShape3: input.shape.length > 3 ? input.shape[3] : 1,
                    inputStride3: input.shape.length > 3 ? input.strides[3] : 1,
                    outputStride3: outputShape.length > 3 ? outputStrides[3] : 1,
                };
                if (!keepdim) outputShape.splice(dim, 1);
                return input.runKernel("any_dim", {dim,maxdim:inputShape.length,dtype:"float32"}, params, [outputShape])[0];
            } else {
                throw new Error("Multi-dimension reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("any", {"dtype":"float32","workgroupSize":256}, params, [[]])[0];
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        ctx.dim = dim;
        ctx.keepdim = keepdim;
        ctx.saveForBackward(input, output);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, output] = ctx.savedTensors as [Tensor, Tensor];
        const dim: number | number[] | undefined = ctx.dim;
        const keepdim: boolean | undefined = ctx.keepdim;
        if (dim !== undefined) {
            if (typeof dim === "number") {
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("any_dim_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
            } else {
                throw new Error("Multi-dimension backward reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("any_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
        }
    }
}
export class MeanFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        if (dim !== undefined) {
            dim = Array.isArray(dim) && dim.length === 1 ? dim[0] : dim;
            if (typeof dim === "number") {
                const inputShape = input.shape;
                let outputShape = input.shape.slice();
                outputShape[dim] = 1;
                let outputStrides = defaultStrides(outputShape);
                const params = {
                    size: shapeSize(outputShape),
                    inputShape0: input.shape.length > 0 ? input.shape[0] : 1,
                    inputStride0: input.shape.length > 0 ? input.strides[0] : 1,
                    outputStride0: outputShape.length > 0 ? outputStrides[0] : 1,
                    inputShape1: input.shape.length > 1 ? input.shape[1] : 1,
                    inputStride1: input.shape.length > 1 ? input.strides[1] : 1,
                    outputStride1: outputShape.length > 1 ? outputStrides[1] : 1,
                    inputShape2: input.shape.length > 2 ? input.shape[2] : 1,
                    inputStride2: input.shape.length > 2 ? input.strides[2] : 1,
                    outputStride2: outputShape.length > 2 ? outputStrides[2] : 1,
                    inputShape3: input.shape.length > 3 ? input.shape[3] : 1,
                    inputStride3: input.shape.length > 3 ? input.strides[3] : 1,
                    outputStride3: outputShape.length > 3 ? outputStrides[3] : 1,
                };
                if (!keepdim) outputShape.splice(dim, 1);
                return input.runKernel("mean_dim", {dim,maxdim:inputShape.length,dtype:"float32"}, params, [outputShape])[0];
            } else {
                throw new Error("Multi-dimension reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("mean", {"dtype":"float32","workgroupSize":256}, params, [[]])[0];
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        ctx.dim = dim;
        ctx.keepdim = keepdim;
        ctx.saveForBackward(input, output);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, output] = ctx.savedTensors as [Tensor, Tensor];
        const dim: number | number[] | undefined = ctx.dim;
        const keepdim: boolean | undefined = ctx.keepdim;
        if (dim !== undefined) {
            if (typeof dim === "number") {
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("mean_dim_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
            } else {
                throw new Error("Multi-dimension backward reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("mean_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
        }
    }
}
export class NormFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        if (dim !== undefined) {
            dim = Array.isArray(dim) && dim.length === 1 ? dim[0] : dim;
            if (typeof dim === "number") {
                const inputShape = input.shape;
                let outputShape = input.shape.slice();
                outputShape[dim] = 1;
                let outputStrides = defaultStrides(outputShape);
                const params = {
                    size: shapeSize(outputShape),
                    inputShape0: input.shape.length > 0 ? input.shape[0] : 1,
                    inputStride0: input.shape.length > 0 ? input.strides[0] : 1,
                    outputStride0: outputShape.length > 0 ? outputStrides[0] : 1,
                    inputShape1: input.shape.length > 1 ? input.shape[1] : 1,
                    inputStride1: input.shape.length > 1 ? input.strides[1] : 1,
                    outputStride1: outputShape.length > 1 ? outputStrides[1] : 1,
                    inputShape2: input.shape.length > 2 ? input.shape[2] : 1,
                    inputStride2: input.shape.length > 2 ? input.strides[2] : 1,
                    outputStride2: outputShape.length > 2 ? outputStrides[2] : 1,
                    inputShape3: input.shape.length > 3 ? input.shape[3] : 1,
                    inputStride3: input.shape.length > 3 ? input.strides[3] : 1,
                    outputStride3: outputShape.length > 3 ? outputStrides[3] : 1,
                };
                if (!keepdim) outputShape.splice(dim, 1);
                return input.runKernel("norm_dim", {dim,maxdim:inputShape.length,dtype:"float32"}, params, [outputShape])[0];
            } else {
                throw new Error("Multi-dimension reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("norm", {"dtype":"float32","workgroupSize":256}, params, [[]])[0];
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        ctx.dim = dim;
        ctx.keepdim = keepdim;
        ctx.saveForBackward(input, output);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, output] = ctx.savedTensors as [Tensor, Tensor];
        const dim: number | number[] | undefined = ctx.dim;
        const keepdim: boolean | undefined = ctx.keepdim;
        if (dim !== undefined) {
            if (typeof dim === "number") {
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("norm_dim_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
            } else {
                throw new Error("Multi-dimension backward reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("norm_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
        }
    }
}
export class ProdFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        if (dim !== undefined) {
            dim = Array.isArray(dim) && dim.length === 1 ? dim[0] : dim;
            if (typeof dim === "number") {
                const inputShape = input.shape;
                let outputShape = input.shape.slice();
                outputShape[dim] = 1;
                let outputStrides = defaultStrides(outputShape);
                const params = {
                    size: shapeSize(outputShape),
                    inputShape0: input.shape.length > 0 ? input.shape[0] : 1,
                    inputStride0: input.shape.length > 0 ? input.strides[0] : 1,
                    outputStride0: outputShape.length > 0 ? outputStrides[0] : 1,
                    inputShape1: input.shape.length > 1 ? input.shape[1] : 1,
                    inputStride1: input.shape.length > 1 ? input.strides[1] : 1,
                    outputStride1: outputShape.length > 1 ? outputStrides[1] : 1,
                    inputShape2: input.shape.length > 2 ? input.shape[2] : 1,
                    inputStride2: input.shape.length > 2 ? input.strides[2] : 1,
                    outputStride2: outputShape.length > 2 ? outputStrides[2] : 1,
                    inputShape3: input.shape.length > 3 ? input.shape[3] : 1,
                    inputStride3: input.shape.length > 3 ? input.strides[3] : 1,
                    outputStride3: outputShape.length > 3 ? outputStrides[3] : 1,
                };
                if (!keepdim) outputShape.splice(dim, 1);
                return input.runKernel("prod_dim", {dim,maxdim:inputShape.length,dtype:"float32"}, params, [outputShape])[0];
            } else {
                throw new Error("Multi-dimension reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("prod", {"dtype":"float32","workgroupSize":256}, params, [[]])[0];
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        ctx.dim = dim;
        ctx.keepdim = keepdim;
        ctx.saveForBackward(input, output);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, output] = ctx.savedTensors as [Tensor, Tensor];
        const dim: number | number[] | undefined = ctx.dim;
        const keepdim: boolean | undefined = ctx.keepdim;
        if (dim !== undefined) {
            if (typeof dim === "number") {
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("prod_dim_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
            } else {
                throw new Error("Multi-dimension backward reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("prod_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
        }
    }
}
export class SumFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        if (dim !== undefined) {
            dim = Array.isArray(dim) && dim.length === 1 ? dim[0] : dim;
            if (typeof dim === "number") {
                const inputShape = input.shape;
                let outputShape = input.shape.slice();
                outputShape[dim] = 1;
                let outputStrides = defaultStrides(outputShape);
                const params = {
                    size: shapeSize(outputShape),
                    inputShape0: input.shape.length > 0 ? input.shape[0] : 1,
                    inputStride0: input.shape.length > 0 ? input.strides[0] : 1,
                    outputStride0: outputShape.length > 0 ? outputStrides[0] : 1,
                    inputShape1: input.shape.length > 1 ? input.shape[1] : 1,
                    inputStride1: input.shape.length > 1 ? input.strides[1] : 1,
                    outputStride1: outputShape.length > 1 ? outputStrides[1] : 1,
                    inputShape2: input.shape.length > 2 ? input.shape[2] : 1,
                    inputStride2: input.shape.length > 2 ? input.strides[2] : 1,
                    outputStride2: outputShape.length > 2 ? outputStrides[2] : 1,
                    inputShape3: input.shape.length > 3 ? input.shape[3] : 1,
                    inputStride3: input.shape.length > 3 ? input.strides[3] : 1,
                    outputStride3: outputShape.length > 3 ? outputStrides[3] : 1,
                };
                if (!keepdim) outputShape.splice(dim, 1);
                return input.runKernel("sum_dim", {dim,maxdim:inputShape.length,dtype:"float32"}, params, [outputShape])[0];
            } else {
                throw new Error("Multi-dimension reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("sum", {"dtype":"float32","workgroupSize":256}, params, [[]])[0];
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        ctx.dim = dim;
        ctx.keepdim = keepdim;
        ctx.saveForBackward(input, output);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, output] = ctx.savedTensors as [Tensor, Tensor];
        const dim: number | number[] | undefined = ctx.dim;
        const keepdim: boolean | undefined = ctx.keepdim;
        if (dim !== undefined) {
            if (typeof dim === "number") {
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("sum_dim_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
            } else {
                throw new Error("Multi-dimension backward reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("sum_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
        }
    }
}
export class CountNonzeroFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        if (dim !== undefined) {
            dim = Array.isArray(dim) && dim.length === 1 ? dim[0] : dim;
            if (typeof dim === "number") {
                const inputShape = input.shape;
                let outputShape = input.shape.slice();
                outputShape[dim] = 1;
                let outputStrides = defaultStrides(outputShape);
                const params = {
                    size: shapeSize(outputShape),
                    inputShape0: input.shape.length > 0 ? input.shape[0] : 1,
                    inputStride0: input.shape.length > 0 ? input.strides[0] : 1,
                    outputStride0: outputShape.length > 0 ? outputStrides[0] : 1,
                    inputShape1: input.shape.length > 1 ? input.shape[1] : 1,
                    inputStride1: input.shape.length > 1 ? input.strides[1] : 1,
                    outputStride1: outputShape.length > 1 ? outputStrides[1] : 1,
                    inputShape2: input.shape.length > 2 ? input.shape[2] : 1,
                    inputStride2: input.shape.length > 2 ? input.strides[2] : 1,
                    outputStride2: outputShape.length > 2 ? outputStrides[2] : 1,
                    inputShape3: input.shape.length > 3 ? input.shape[3] : 1,
                    inputStride3: input.shape.length > 3 ? input.strides[3] : 1,
                    outputStride3: outputShape.length > 3 ? outputStrides[3] : 1,
                };
                if (!keepdim) outputShape.splice(dim, 1);
                return input.runKernel("countNonzero_dim", {dim,maxdim:inputShape.length,dtype:"float32"}, params, [outputShape])[0];
            } else {
                throw new Error("Multi-dimension reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("countNonzero", {"dtype":"float32","workgroupSize":256}, params, [[]])[0];
        }
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        let [input, dim, keepdim] = inputs as [Tensor, number | number[] | undefined, boolean | undefined];
        ctx.dim = dim;
        ctx.keepdim = keepdim;
        ctx.saveForBackward(input, output);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, output] = ctx.savedTensors as [Tensor, Tensor];
        const dim: number | number[] | undefined = ctx.dim;
        const keepdim: boolean | undefined = ctx.keepdim;
        if (dim !== undefined) {
            if (typeof dim === "number") {
                const params = {
                    size: shapeSize(input.shape),
                };
                return input.runKernel("countNonzero_dim_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
            } else {
                throw new Error("Multi-dimension backward reduction not supported");
            }
        } else {
            const params = {
                size: shapeSize(input.shape),
            };
            return input.runKernel("countNonzero_grad", {"dtype":"float32","workgroupSize":256}, params, [input.shape], output, outputGrad);
        }
    }
}
