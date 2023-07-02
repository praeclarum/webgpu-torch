
import { SGD, required } from "./optim";
import { Tensor } from "./tensor";
import { sigmoid } from "./ops_opgen";
import { Sequential } from "./nn_module";
import { Linear } from "./nn_basic";
import { ReLU } from "./nn_opgen";
import { ones } from "./factories";
import { noGrad } from "./autograd";

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
    const batchSize = 17;
    const hiddenSize = 31;
    const maxSteps = 50;
    const printInterval = 1000;
    const model = new Sequential(
        new Linear(3, hiddenSize),
        new ReLU(),
        new Linear(hiddenSize, hiddenSize),
        new ReLU(),
        new Linear(hiddenSize, 1),
    );
    const radius = ones([batchSize, 1]);
    const sphereSDF = (batchedPoints: Tensor) => {
        // sqrt(x^2 + y^2 + z^2) - radius
        // const distanceToCenterSq = batchedPoints.pow(2).sum(1, true);
        const squaredPoint = batchedPoints.pow(2);
        const distanceToCenterSq = squaredPoint.sum(1, true);
        const distanceToCenter = distanceToCenterSq.sqrt();
        const distanceToSurface = distanceToCenter.sub(radius);
        return distanceToSurface;
    }
    const sphereA = (batchedPoints: number[][]) => {
        const result: number[] = [];
        for (let i = 0; i < batchedPoints.length; i++) {
            const [x, y, z] = batchedPoints[i];
            const d = Math.sqrt(x*x + y*y + z*z) - 1;
            result.push(d);
        }
        return result;
    }
    const getBatch = () => {
        const pointsArray: number[][] = [];
        for (let batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            pointsArray.push([(Math.random()-0.5)*2.1, (Math.random()-0.5)*2.1, (Math.random()-0.5)*2.1]);
        }
        const points = new Tensor(pointsArray);
        return points;
    };
    const sample = async () => {
        const [e, p] = noGrad(() => {
            const points = getBatch();
            const expectedDistances = sphereSDF(points);
            const predictedDistances = model.forward(points);
            return [expectedDistances, predictedDistances];
        });
        console.log("expectedDistances", await e.toArrayAsync());
        console.log("predictedDistances", await p.toArrayAsync());
    };
    const optimizer = new SGD(Array.from(model.parameters()), 0.01);
    for (let stepIndex = 0; stepIndex < maxSteps; stepIndex++) {
        const points = getBatch();
        const expectedDistances = sphereSDF(points);
        const predictedDistances = model.forward(points);
        const error = predictedDistances.sub(expectedDistances);
        const loss = error.pow(2).mean();
        if (stepIndex > 0 && stepIndex % printInterval === 0) {
            console.log("step", `${stepIndex}/${maxSteps}`, "loss", await loss.toArrayAsync());
        }
        loss.backward();
        optimizer.step();
        optimizer.zeroGrad();
    }
    // sample();
});

