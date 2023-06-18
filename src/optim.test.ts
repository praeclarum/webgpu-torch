
import { SGD, required } from "./optim";
import { Tensor } from "./tensor";
import { sigmoid } from "./ops_opgen";

test("optimizer ctor with empty param list", () => {
    expect(() => new SGD([], 0.1)).toThrow();
});

test("optimizer ctor with param group", () => {
    const params = [new Tensor({data: [1, 2, 3], requiresGrad: true})];
    const optimizer = new SGD({params: params}, 0.1);
    expect(optimizer.paramGroups.length).toBe(1);
});

test("sgd lr required", () => {
    const params = [new Tensor({data: [1, 2, 3], requiresGrad: true})];
    expect(() => new SGD({params: params})).toThrow();
});

test("sgd default params", () => {
    const params = [new Tensor({data: [1, 2, 3], requiresGrad: true})];
    const optimizer = new SGD(params, 0.1);
    expect(optimizer.paramGroups.length).toBe(1);
    expect(optimizer.paramGroups[0].params).toEqual(params);
    expect(optimizer.paramGroups[0].lr).toBe(0.1);
    expect(optimizer.paramGroups[0].momentum).toBe(0);
    expect(optimizer.paramGroups[0].dampening).toBe(0);
    expect(optimizer.paramGroups[0].weightDecay).toBe(0);
    expect(optimizer.paramGroups[0].nesterov).toBe(false);
    expect(optimizer.paramGroups[0].maximize).toBe(false);
    expect(optimizer.paramGroups[0].foreach).toBe(false);
    expect(optimizer.paramGroups[0].differentiable).toBe(false);
});

test("sgd step closure", async () => {
    const params = [new Tensor({data: [1, 2, 3], requiresGrad: true})];
    const optimizer = new SGD(params, 0.1);
    const loss = optimizer.step(() => {
        return new Tensor({data: [10, 20, 30], requiresGrad: true});
    });
    expect(loss).toBeInstanceOf(Tensor);
    const lossValues = await loss?.toArrayAsync();
    expect(lossValues).toEqual([10, 20, 30]);
});

test("sgd step with params with grads", async () => {
    const paramInitialValue = [1.5];
    const params = [new Tensor({data: paramInitialValue, requiresGrad: true})];
    const loss = sigmoid(params[0]);
    const lossValues = await loss.toArrayAsync();
    // console.log("lossValues", lossValues);
    loss.backward();
    const optimizer = new SGD(params, 0.1);
    optimizer.step();
    const paramValue = await params[0].toArrayAsync();
    expect(paramValue).toEqual([1.5 - 0.1 * 0.25]);
});


