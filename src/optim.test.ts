
import { SGD, required } from "./optim";
import { Tensor } from "./tensor";
import { sigmoid } from "./ops_opgen";
import { Sequential } from "./nn_module";
import { Linear } from "./nn_basic";
import { ReLU } from "./nn_opgen";
import { ones } from "./factories";

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
    const lossValues = await loss.toArrayAsync() as number[];
    expect(lossValues[0]).toBeCloseTo(0.81757448);
    loss.backward();
    expect(params[0].grad).not.toBeNull();
    const paramGradValues = await params[0].grad!.toArrayAsync() as number[];
    expect(paramGradValues[0]).toBeCloseTo(0.14914645);
    const optimizer = new SGD(params, 0.1);
    optimizer.step();
    const paramValue = await params[0].toArrayAsync() as number[];
    expect(paramValue[0]).toBeCloseTo(1.5 - 0.1 * 0.14914645);
});

test("sgd mlp train loop", async () => {
    const hiddenSize = 31;
    const model = new Sequential(
        new Linear(3, hiddenSize),
        new ReLU(),
        new Linear(hiddenSize, hiddenSize),
        new ReLU(),
        new Linear(hiddenSize, 1),
    );
    const batchSize = 2;
    const radius = ones([batchSize, 1]);
    const two = ones([batchSize, 1]).mul(2);
    const sphere = (batchedPoints: Tensor) => 
        batchedPoints.pow(2).sum(1, true).sqrt().sub(radius);
    const sphereA = (batchedPoints: number[][]) => {
        const result: number[] = [];
        for (let i = 0; i < batchedPoints.length; i++) {
            const [x, y, z] = batchedPoints[i];
            const d = Math.sqrt(x*x + y*y + z*z) - 1;
            result.push(d);
        }
        return result;
    }
    const maxSteps = 2;
    const optimizer = new SGD(Array.from(model.parameters()), 0.1);
    for (let stepIndex = 0; stepIndex < maxSteps; stepIndex++) {
        const pointsArray: number[][] = [];
        for (let batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            pointsArray.push([(Math.random()-0.5)*4, (Math.random()-0.5)*4, (Math.random()-0.5)*4]);
        }
        const points = new Tensor(pointsArray);
        const expectedDistances = sphere(points);
        // console.log("expectedDistances", await expectedDistances.toArrayAsync());
        // const expectedDistancesA = sphereA(pointsArray);
        // console.log("expectedDistancesA", expectedDistancesA);
        const predictedDistances = model.forward(points);
        // console.log("predictedDistances", await predictedDistances.toArrayAsync());
        const loss = predictedDistances.sub(expectedDistances).pow(two).mean();
        console.log("loss", await loss.toArrayAsync());
        loss.backward();
        optimizer.step();
        // optimizer.zeroGrad();
    }
});

