import { Tensor } from "./tensor";

export type FunctionInput = Tensor | number | boolean | string;
export type FunctionBackwardOutput = Tensor | null;

export class AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Do not call forward on AutoFunction directly.");
    }
    static setupContext(ctx: AutoContext, inputs: FunctionInput[], output: Tensor) {
        throw new Error("Do not call setupContext on AutoFunction directly.");
    }
    static backward(ctx: AutoContext, gradOutput: Tensor): FunctionBackwardOutput[] {
        throw new Error("Do not call backward on AutoFunction directly.");
    }
    static apply(...inputs: FunctionInput[]): Tensor {
        const output = this.forward(...inputs);
        const ctx = new AutoContext(inputs);
        this.setupContext(ctx, inputs, output);
        return output;
    }
}

export class AutoContext {
    savedTensors: Tensor[] = [];
    needsInputGradient: boolean[];
    constructor(inputs: FunctionInput[]) {
        this.needsInputGradient = inputs.map(input => input instanceof Tensor && input.requiresGrad);
    }
    saveForBackward(...tensors: Tensor[]) {
        this.savedTensors = tensors;
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
    static setupContext(ctx: AutoContext, inputs: FunctionInput[], output: Tensor) {
        const [input, weight, bias] = inputs as [Tensor, Tensor, Tensor];
        ctx.saveForBackward(input, weight, bias);
    }
    static backward(ctx: AutoContext, gradOutput: Tensor): FunctionBackwardOutput[] {
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
