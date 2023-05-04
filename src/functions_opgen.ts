import {
    AutoFunction,
    FunctionInput,
    GradientFunctionContext,
    GradientFunctionOutput,
} from "./autograd";
import { Tensor } from "./tensor";
import * as ops from "./ops";
export class AbsFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        return ops.abs(input);
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AcosFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        return ops.acos(input);
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AcoshFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        return ops.acosh(input);
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AddFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        return ops.add(input, other, alpha);
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AsinFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        return ops.asin(input);
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AsinhFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        return ops.asinh(input);
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AtanFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        return ops.atan(input);
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class Atan2Function extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input, other] = inputs as [Tensor, Tensor];
        return ops.atan2(input, other);
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class SubFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input, other, alpha] = inputs as [Tensor, Tensor, number|undefined];
        return ops.sub(input, other, alpha);
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
