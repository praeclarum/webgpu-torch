import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import { matmul } from "./ops_artisanal";
import { Shape } from "./shape";
import type { Tensor } from "./tensor";

function _grad_sum_to_size(grad: Tensor, targetShape: Shape): Tensor {
    const gradShape = grad.shape;
    let sumDims: number[] = [];
    let viewShape: Shape = [];

    for (let i = 0; i < gradShape.length; i++) {
        if (gradShape[i] === targetShape[i]) {
            viewShape.push(1);
        } else {
            viewShape.push(gradShape[i]);
            sumDims.push(i);
        }
    }
    return grad.sum(sumDims[0], true);
    // let reshapedGrad = grad.view(viewShape);
    // reshapedGrad = reshapedGrad.sum(sumDims, true);
    // return reshapedGrad.view(targetShape);
}

export class LinearFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, weight, bias] = inputs as [Tensor, Tensor, Tensor?];
        if (bias !== undefined) {
            if (bias.shape.length !== 1) {
                throw new Error("bias must be 1D");
            }
            if (bias.shape[0] !== weight.shape[0]) {
                throw new Error("bias must have same size as weight");
            }
        }
        const output = matmul(input, weight.t());
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
            inputGrad = matmul(gradOutput, weight);
        }
        if (ctx.needsInputGradient[1]) {
            weightGrad = matmul(gradOutput.t(), input);
        }
        if (ctx.needsInputGradient[2]) {
            biasGrad = _grad_sum_to_size(gradOutput, bias.shape);
        }
        return [inputGrad, weightGrad, biasGrad];
    }
}
