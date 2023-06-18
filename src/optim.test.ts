
import { SGD, required } from "./optim";
import { Tensor } from "./tensor";

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

