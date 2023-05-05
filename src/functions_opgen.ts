import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import { Tensor } from "./tensor";
import { shapeSize } from "./shape";
import * as ops from "./ops";
export class AbsFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("abs", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("absGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AbsGradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("absGrad", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("absGradGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AcosFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acos", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("acosGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AcosGradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acosGrad", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("acosGradGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AcoshFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acosh", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("acoshGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AcoshGradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acoshGrad", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("acoshGradGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AddFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        const params = {
            size: shapeSize(input.shape),
            alpha: alpha || 1.0,
        };
        return input.runKernel("add", { dtype: input.dtype }, params, [input.shape], other)[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        ctx.alpha = alpha;
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        const params = {
            size: shapeSize(input.shape),
            alpha: ctx.alpha || 1.0,
        };
        return input.runKernel("addGrad", { dtype: input.dtype }, params, [input.shape, other.shape], other, outputGrad);
    }
}
export class AddGradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        const params = {
            size: shapeSize(input.shape),
            alpha: alpha || 1.0,
        };
        return input.runKernel("addGrad", { dtype: input.dtype }, params, [input.shape], other)[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        ctx.alpha = alpha;
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        const params = {
            size: shapeSize(input.shape),
            alpha: ctx.alpha || 1.0,
        };
        return input.runKernel("addGradGrad", { dtype: input.dtype }, params, [input.shape, other.shape], other, outputGrad);
    }
}
export class AsinFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asin", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("asinGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AsinGradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asinGrad", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("asinGradGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AsinhFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asinh", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("asinhGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AsinhGradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asinhGrad", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("asinhGradGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AtanFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atan", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("atanGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class AtanGradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atanGrad", { dtype: input.dtype }, params, [input.shape])[0];
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
        return input.runKernel("atanGradGrad", { dtype: input.dtype }, params, [input.shape], outputGrad);
    }
}
export class Atan2Function extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atan2", { dtype: input.dtype }, params, [input.shape], other)[0];
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
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atan2Grad", { dtype: input.dtype }, params, [input.shape, other.shape], other, outputGrad);
    }
}
export class Atan2GradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atan2Grad", { dtype: input.dtype }, params, [input.shape], other)[0];
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
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atan2GradGrad", { dtype: input.dtype }, params, [input.shape, other.shape], other, outputGrad);
    }
}
export class MulFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        const params = {
            size: shapeSize(input.shape),
            alpha: alpha || 1.0,
        };
        return input.runKernel("mul", { dtype: input.dtype }, params, [input.shape], other)[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        ctx.alpha = alpha;
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        const params = {
            size: shapeSize(input.shape),
            alpha: ctx.alpha || 1.0,
        };
        return input.runKernel("mulGrad", { dtype: input.dtype }, params, [input.shape, other.shape], other, outputGrad);
    }
}
export class MulGradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        const params = {
            size: shapeSize(input.shape),
            alpha: alpha || 1.0,
        };
        return input.runKernel("mulGrad", { dtype: input.dtype }, params, [input.shape], other)[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        ctx.alpha = alpha;
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        const params = {
            size: shapeSize(input.shape),
            alpha: ctx.alpha || 1.0,
        };
        return input.runKernel("mulGradGrad", { dtype: input.dtype }, params, [input.shape, other.shape], other, outputGrad);
    }
}
export class SubFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        const params = {
            size: shapeSize(input.shape),
            alpha: alpha || 1.0,
        };
        return input.runKernel("sub", { dtype: input.dtype }, params, [input.shape], other)[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        ctx.alpha = alpha;
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        const params = {
            size: shapeSize(input.shape),
            alpha: ctx.alpha || 1.0,
        };
        return input.runKernel("subGrad", { dtype: input.dtype }, params, [input.shape, other.shape], other, outputGrad);
    }
}
export class SubGradFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        const params = {
            size: shapeSize(input.shape),
            alpha: alpha || 1.0,
        };
        return input.runKernel("subGrad", { dtype: input.dtype }, params, [input.shape], other)[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        ctx.alpha = alpha;
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {
        const [input, other] = ctx.savedTensors as [Tensor, Tensor];
        const params = {
            size: shapeSize(input.shape),
            alpha: ctx.alpha || 1.0,
        };
        return input.runKernel("subGradGrad", { dtype: input.dtype }, params, [input.shape, other.shape], other, outputGrad);
    }
}
