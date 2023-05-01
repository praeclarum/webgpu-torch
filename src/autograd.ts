import {
    Tensor,
    GradientFunctionContext,
    FunctionInput,
    GradientFunctionOutput,
} from "./tensor";

export class AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Do not call forward on AutoFunction directly.");
    }
    static setupContext(
        ctx: GradientFunctionContext,
        inputs: FunctionInput[],
        output: Tensor
    ) {
        throw new Error("Do not call setupContext on AutoFunction directly.");
    }
    static backward(
        ctx: GradientFunctionContext,
        gradOutput: Tensor
    ): GradientFunctionOutput[] {
        throw new Error("Do not call backward on AutoFunction directly.");
    }
    static apply(...inputs: FunctionInput[]): Tensor {
        const ctx = new GradientFunctionContext(inputs);
        const detachedInputs = inputs.map((input) =>
            input instanceof Tensor ? input.detach() : input
        );
        const output = this.forward(...detachedInputs);
        this.setupContext(ctx, detachedInputs, output);
        output.setGradientFunction(ctx, this.backward);
        return output;
    }
}

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
        ctx: GradientFunctionContext,
        inputs: FunctionInput[],
        output: Tensor
    ) {
        const [input, weight, bias] = inputs as [Tensor, Tensor, Tensor];
        ctx.saveForBackward(input, weight, bias);
    }
    static backward(
        ctx: GradientFunctionContext,
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
            gradWeight = gradOutput.t().mm(input);
        }
        if (bias && ctx.needsInputGradient[2]) {
            gradBias = gradOutput.sum(0);
        }
        return [gradInput, gradWeight, gradBias];
    }
}
