import { Tensor } from "./tensor";
import { AutoFunction, GradientContext } from "./autograd";
import { zeros } from "./factories";
import { linear } from "./ops_artisanal";
import { Sequential } from "./nn_module";
import { Linear } from "./nn_basic";
import { ReLU, Sigmoid } from "./nn_opgen";

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
    const loss = output.sum();
    expect(loss.gradFunc).not.toBeNull();
    expect(loss.requiresGrad).toBe(true);
    loss.backward();
    expect(await loss.toArrayAsync()).toEqual(461);
    expect(weight.grad).not.toBeNull();
    expect(weight.grad!.shape).toEqual([2, 3]);
    expect(bias.grad).not.toBeNull();
    expect(bias.grad!.shape).toEqual([2]);
    expect(await weight.grad?.toArrayAsync()).toEqual([[5., 7., 9.], [5., 7., 9.]]);
    expect(await bias.grad?.toArrayAsync()).toEqual([2., 2.]);
});

test("mlp backwards", async () => {
    const linear1 = new Linear(2, 3);
    const linear2 = new Linear(3, 5);
    const model = new Sequential(
        linear1,
        new ReLU(),
        linear2,
        new Sigmoid()
    );
    const input = new Tensor({data:[[-0.1, 0.2], [0.3, -0.4]], requiresGrad: false});
    const output = model.forward(input);
    const outputArray = await output.toArrayAsync();
    const loss = output.sum();
    expect(loss.gradFunc).not.toBeNull();
    expect(loss.requiresGrad).toBe(true);
    loss.backward();
    const l1wgrad = linear1.weight.grad!;
    expect(l1wgrad).not.toBeNull();
    expect(l1wgrad.shape).toEqual([3, 2]);
    const l1bgrad = linear1.bias!.grad!;
    expect(l1bgrad).not.toBeNull();
    expect(l1bgrad.shape).toEqual([3]);
});
