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
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("abs", { dtype: input.dtype }, params);
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AcosFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acos", { dtype: input.dtype }, params);
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AcoshFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("acosh", { dtype: input.dtype }, params);
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AddFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        const params = {
            size: shapeSize(input.shape),
            alpha: alpha || 1.0,
        };
        return input.runKernel("add", { dtype: input.dtype }, params, other);
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
    static backward(ctx: GradientContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AsinFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asin", { dtype: input.dtype }, params);
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AsinhFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("asinh", { dtype: input.dtype }, params);
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AtanFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atan", { dtype: input.dtype }, params);
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(ctx: GradientContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class Atan2Function extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        const params = {
            size: shapeSize(input.shape),
        };
        return input.runKernel("atan2", { dtype: input.dtype }, params, other);
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const [input, other] = inputs as [Tensor, Tensor];
        ctx.saveForBackward(input, other);
    }
    static backward(ctx: GradientContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class SubFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        const params = {
            size: shapeSize(input.shape),
            alpha: alpha || 1.0,
        };
        return input.runKernel("sub", { dtype: input.dtype }, params, other);
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
    static backward(ctx: GradientContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
