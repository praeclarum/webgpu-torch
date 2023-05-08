import { Tensor } from "./tensor";
import { AutoFunction, GradientContext } from "./autograd";
import { zeros } from "./factories";
import * as functions from "./functions";

test("auto function forward fails", () => {
    expect(() => AutoFunction.forward([])).toThrow();
});
test("auto function setupContext fails", () => {
    expect(() => AutoFunction.setupContext(new GradientContext([]), [zeros(3)], zeros(3))).toThrow();
});
test("auto function backward fails", () => {
    expect(() => AutoFunction.backward(new GradientContext([]), zeros(3))).toThrow();
});

test("linear backwards", () => {
    const input = new Tensor({data:[[3]], requiresGrad: true});
    const weight = new Tensor({data:[[10], [11]], requiresGrad:  true});
    const bias = new Tensor({data:[[1000, 10000]], requiresGrad:  true});
    const output = functions.LinearFunction.apply(input, weight, bias);
    expect(output).toBeInstanceOf(Tensor);
    expect(output.shape).toEqual([1, 2]);
    const loss = output.sum();
    expect(loss.gradFunc).not.toBeNull();
    expect(loss.requiresGrad).toBe(true);
    // loss.backward();
    // expect(input.grad).not.toBeNull();
    // expect(weight.grad).not.toBeNull();
    // expect(bias.grad).not.toBeNull();
    // expect(output.grad).not.toBeNull();
});
