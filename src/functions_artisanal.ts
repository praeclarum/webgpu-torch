import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import type { Tensor } from "./tensor";

export class LinearFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
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
        let inputGrad: Tensor | null = null;
        let weightGrad: Tensor | null = null;
        let biasGrad: Tensor | null = null;
        if (ctx.needsInputGradient[0]) {
            inputGrad = gradOutput.mm(weight);
        }
        if (ctx.needsInputGradient[1]) {
            weightGrad = input.t().mm(gradOutput);
        }
        if (ctx.needsInputGradient[2]) {
            biasGrad = gradOutput.sum(0);
        }
        return [inputGrad, weightGrad, biasGrad];
    }
}
