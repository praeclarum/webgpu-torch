import { Tensor } from "./tensor";
import { AutoFunction, GradientContext } from "./autograd";
import { zeros } from "./factories";
import { linear } from "./ops_artisanal";

test("auto function forward fails", () => {
    expect(() => AutoFunction.forward([])).toThrow();
});
test("auto function setupContext fails", () => {
    expect(() => AutoFunction.setupContext(new GradientContext([]), [zeros(3)], zeros(3))).toThrow();
});
test("auto function backward fails", () => {
    expect(() => AutoFunction.backward(new GradientContext([]), zeros(3))).toThrow();
});

test("linear backwards", async () => {
    const input = new Tensor({data:[[3]], requiresGrad: false});
    const weight = new Tensor({data:[[10], [11]], requiresGrad:  true});
    const bias = new Tensor({data:[[1000, 10000]], requiresGrad:  true});
    const output = linear(input, weight, bias);
    expect(await output.toArrayAsync()).toEqual([[1030, 10033]]);
    const loss = output.sum();
    expect(loss.gradFunc).not.toBeNull();
    expect(loss.requiresGrad).toBe(true);
    loss.backward();
    expect(await loss.toArrayAsync()).toEqual(11063);
    // expect(weight.grad).not.toBeNull();
    // expect(weight.grad!.shape).toEqual([2, 1]);
    // expect(bias.grad).not.toBeNull();
    // expect(bias.grad!.shape).toEqual([1, 2]);
    // expect(await weight.grad?.toArrayAsync()).toEqual([[3], [3]]);
    // expect(await bias.grad?.toArrayAsync()).toEqual([[1, 1]]);
});
