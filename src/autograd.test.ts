import { Tensor, GradientFunctionContext } from "./tensor";
import { AutoFunction, LinearFunction } from "./autograd";
import { zeros } from "./index";

test("auto function forward fails", () => {
    expect(() => AutoFunction.forward()).toThrow();
});
test("auto function setupContext fails", () => {
    expect(() => AutoFunction.setupContext(new GradientFunctionContext([]), [zeros(3)], zeros(3))).toThrow();
});
test("auto function backward fails", () => {
    expect(() => AutoFunction.backward(new GradientFunctionContext([]), zeros(3))).toThrow();
});

test("linear backwards", () => {
    const input = new Tensor([[3]], "float32", null, true);
    const weight = new Tensor([[10], [11]], "float32", null, true);
    const bias = new Tensor([[1000, 10000]], "float32", null, true);
    const output = LinearFunction.apply(input, weight, bias);
    expect(output).toBeInstanceOf(Tensor);
    expect(output.shape).toEqual([1, 2]);
    const loss = output.sum();
    expect(loss.gradFunc).not.toBeNull();
    expect(loss.requiresGrad).toBe(true);
    loss.backward();
    expect(input.grad).not.toBeNull();
    expect(weight.grad).not.toBeNull();
    expect(bias.grad).not.toBeNull();
    expect(output.grad).not.toBeNull();
});
