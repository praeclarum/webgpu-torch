import {
    AutoFunction,
    FunctionInput,
    GradientFunctionContext,
    GradientFunctionOutput,
} from "./autograd";
import { Tensor } from "./tensor";
export class AbsFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Not implemented");
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AcosFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Not implemented");
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AcoshFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Not implemented");
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AddFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Not implemented");
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AsinFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Not implemented");
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AsinhFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Not implemented");
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class AtanFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Not implemented");
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class Atan2Function extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Not implemented");
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
export class SubFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Not implemented");
    }
    static backward(ctx: GradientFunctionContext, gradOutput: Tensor): GradientFunctionOutput[] {
        throw new Error("Not implemented");
    }
}
