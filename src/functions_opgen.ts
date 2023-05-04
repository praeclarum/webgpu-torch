import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import { Tensor } from "./tensor";
import * as ops from "./ops";
export class AbsFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        return ops.abs(input);
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
        return ops.acos(input);
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
        return ops.acosh(input);
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
        return ops.add(input, other, alpha);
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
        return ops.asin(input);
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
        return ops.asinh(input);
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
        return ops.atan(input);
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
        return ops.atan2(input, other);
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
        return ops.sub(input, other, alpha);
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
