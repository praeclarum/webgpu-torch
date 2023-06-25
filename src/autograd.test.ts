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
    const input = new Tensor({data:[[1, 2, 3], [4, 5, 6]], requiresGrad: false});
    const weight = new Tensor({data:[[7, 8, 9], [10, 11, 12]], requiresGrad:  true});
    const bias = new Tensor({data:[13, 14], requiresGrad:  true});
    const output = linear(input, weight, bias);
    expect(await output.toArrayAsync()).toEqual([[ 63.,  82.], [135., 181.]]);
    // const loss = output.sum();
    // expect(loss.gradFunc).not.toBeNull();
    // expect(loss.requiresGrad).toBe(true);
    // loss.backward();
    // expect(await loss.toArrayAsync()).toEqual(11063);
    // expect(weight.grad).not.toBeNull();
    // expect(weight.grad!.shape).toEqual([2, 3]);
    // expect(bias.grad).not.toBeNull();
    // expect(bias.grad!.shape).toEqual([2]);
    // expect(await weight.grad?.toArrayAsync()).toEqual([[5., 7., 9.],
    //     [5., 7., 9.]]);
    // expect(await bias.grad?.toArrayAsync()).toEqual([2., 2.]);
});
