import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import { Tensor } from "./tensor";

export class LinearFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input, weight, bias] = inputs as [Tensor, Tensor, Tensor];
        const output = input.mm(weight.t());
        if (bias) {
            output.add_(bias);
        }
        return output;
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ) {
        const [input, weight, bias] = inputs as [Tensor, Tensor, Tensor];
        ctx.saveForBackward(input, weight, bias);
    }
    static backward(
        ctx: GradientContext,
        gradOutput: Tensor
    ): GradientFunctionOutput[] {
        const [input, weight, bias] = ctx.savedTensors;
        let gradInput: Tensor | null = null;
        let gradWeight: Tensor | null = null;
        let gradBias: Tensor | null = null;
        if (ctx.needsInputGradient[0]) {
            gradInput = gradOutput.mm(weight);
        }
        if (ctx.needsInputGradient[1]) {
            gradWeight = input.t().mm(gradOutput);
        }
        if (ctx.needsInputGradient[2]) {
            gradBias = gradOutput.sum(0);
        }
        return [gradInput, gradWeight, gradBias];
    }
}

export class SumAxisFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Method not implemented.");
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ) {
        throw new Error("Method not implemented.");
    }
    static backward(
        ctx: GradientContext,
        gradOutput: Tensor
    ): GradientFunctionOutput[] {
        throw new Error("Method not implemented.");
    }
}

export class SumFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        const [input] = inputs as [Tensor];
        return input.sum();
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ) {
        const [input] = inputs as [Tensor];
        ctx.saveForBackward(input);
    }
    static backward(
        ctx: GradientContext,
        gradOutput: Tensor
    ): GradientFunctionOutput[] {
        const [input] = ctx.savedTensors;
        let gradInput: Tensor | null = null;
        if (ctx.needsInputGradient[0]) {
            gradInput = gradOutput.expand(input.shape);
        }
        return [gradInput];
    }
}
