import { ones } from "./factories";
import { tensor } from "./ops_artisanal";

test("sigmoid(1).size == 307", async () => {
    // prime number > 300
    const n = 307;
    const x = ones(n);
    const y = x.sigmoid();
    expect(y.shape[0]).toBe(n);
    const a = await y.toArrayAsync() as number[];
    expect(a.length).toBe(n);
    for (let i = 0; i < n; i++) {
        expect(a[i]).toBeCloseTo(0.7310585786300049);
    }
});

test("add with alpha", async () => {
    const x = ones(3);
    const y = x.add(ones(3), 5);
    const a = await y.toArrayAsync();
    expect(a).toEqual([6, 6, 6]);
});

test("sub with alpha", async () => {
    const x = ones(3);
    const y = x.sub(ones(3), 7);
    const a = await y.toArrayAsync();
    expect(a).toEqual([-6, -6, -6]);
});

test("sum vector", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.sum();
    const a = await y.toArrayAsync();
    expect(a).toEqual(6);
});

test("sum vector grad", async () => {
    const x = tensor({data:[1, 2, 3, 4, 5, 6], requiresGrad: true});
    const y = x.sum();
    y.backward();
    const a = await x.grad!.toArrayAsync();
    expect(a).toEqual([1, 1, 1, 1, 1, 1]);
});

test("mean vector", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.mean();
    expect(await y.toArrayAsync()).toBeCloseTo(2);
});

test("mean vector grad", async () => {
    const x = tensor({data:[1, 2, 3, 4], requiresGrad: true});
    const y = x.mean();
    y.backward();
    expect(await x.grad!.toArrayAsync()).toEqual([0.25, 0.25, 0.25, 0.25]);
});

test("add 1d broadcasts", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.add(tensor([4]));
    expect(await y.toArrayAsync()).toEqual([5, 6, 7]);
});

test("add 2d to 1d broadcasts", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    const y = x.add(tensor([4]));
    expect(await y.toArrayAsync()).toEqual([[ 5,  6,  7], [ 8,  9, 10]]);
});

test("add 2d to 2d broadcasts", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    const y = x.add(tensor([[4]]));
    expect(await y.toArrayAsync()).toEqual([[ 5,  6,  7], [ 8,  9, 10]]);
});

test("add 2d to 2d inplace broadcasts", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    x.add_(tensor([[4]]));
    expect(await x.toArrayAsync()).toEqual([[ 5,  6,  7], [ 8,  9, 10]]);
});

test("flatten 2d", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    const y = x.flatten();
    expect(y.shape).toEqual([6]);
    expect(await y.toArrayAsync()).toEqual([1, 2, 3, 4, 5, 6]);
});

// test("gather 1d", async () => {
//     const x = tensor([1, 2, 3]);
//     const y = x.gather(1, tensor([2, 1, 0]));
//     expect(await y.toArrayAsync()).toEqual([3, 2, 1]);
// });

test("numel", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    expect(x.numel()).toEqual(6);
});

test("pow(3) of 31x3 random numbers", async () => {
    const batchSize = 31;
    const pointsArray: number[][] = [];
    for (let batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        pointsArray.push([(Math.random()-0.5)*2.1, (Math.random()-0.5)*2.1, (Math.random()-0.5)*2.1]);
    }
    const points = tensor({data:pointsArray, requiresGrad: true});
    const y = points.pow(3);
    expect(y.shape).toEqual([batchSize, 3]);
    const yArray = await y.toArrayAsync() as number[][];
    for (let b = 0; b < batchSize; b++) {
        for (let i = 0; i < 3; i++) {
            expect(yArray[b][i]).not.toBeNaN();
            expect(yArray[b][i]).toBeCloseTo(pointsArray[b][i] * pointsArray[b][i] * pointsArray[b][i]);
        }
    }
    const loss = y.sum();
    const lossValue = await loss.toArrayAsync() as number;
    expect(lossValue).not.toBeNaN();
    loss.backward();
    expect(points.grad).not.toBeNull();
    const gradArray = await points.grad!.toArrayAsync() as number[][];
    for (let b = 0; b < batchSize; b++) {
        for (let i = 0; i < 3; i++) {
            expect(gradArray[b][i]).not.toBeNaN();
            expect(gradArray[b][i]).toBeCloseTo(3 * pointsArray[b][i] * pointsArray[b][i]);
        }
    }
});

test("pow(2) grad of 1 negative number", async () => {
    const pointsArray: number[] = [-3.14];
    const points = tensor({data:pointsArray, requiresGrad: true});
    const y = points.pow(2);
    const yArray = await y.toArrayAsync() as number[][];
    expect(yArray[0]).not.toBeNaN();
    expect(yArray[0]).toBeCloseTo(pointsArray[0] * pointsArray[0]);
    const loss = y.abs();
    const lossValue = await loss.toArrayAsync() as number;
    expect(lossValue).not.toBeNaN();
    loss.backward();
    expect(points.grad).not.toBeNull();
    expect(y.grad).not.toBeNull();
    const gradArray = await points.grad!.toArrayAsync() as number[][];
    expect(gradArray[0]).not.toBeNaN();
    expect(gradArray[0]).toBeCloseTo(-6.2800);
});

test("pow(3) grad of 1 negative number", async () => {
    const pointsArray: number[] = [-3.14];
    const points = tensor({data:pointsArray, requiresGrad: true});
    const y = points.pow(3);
    const yArray = await y.toArrayAsync() as number[][];
    expect(yArray[0]).not.toBeNaN();
    expect(yArray[0]).toBeCloseTo(pointsArray[0] * pointsArray[0] * pointsArray[0]);
    const loss = y.abs();
    const lossValue = await loss.toArrayAsync() as number;
    expect(lossValue).not.toBeNaN();
    loss.backward();
    expect(points.grad).not.toBeNull();
    expect(y.grad).not.toBeNull();
    const gradArray = await points.grad!.toArrayAsync() as number[][];
    expect(gradArray[0]).not.toBeNaN();
    expect(gradArray[0]).toBeCloseTo(-29.5788);
});

test("pow(2.5) grad of 1 positive number", async () => {
    const pointsArray: number[] = [3.14];
    const points = tensor({data:pointsArray, requiresGrad: true});
    const y = points.pow(2.5);
    const yArray = await y.toArrayAsync() as number[][];
    expect(yArray[0]).not.toBeNaN();
    expect(yArray[0]).toBeCloseTo(Math.pow(pointsArray[0], 2.5));
    const loss = y.abs();
    const lossValue = await loss.toArrayAsync() as number;
    expect(lossValue).not.toBeNaN();
    loss.backward();
    expect(points.grad).not.toBeNull();
    expect(y.grad).not.toBeNull();
    const gradArray = await points.grad!.toArrayAsync() as number[][];
    expect(gradArray[0]).not.toBeNaN();
    expect(gradArray[0]).toBeCloseTo(13.9102);
});

test("reshape [2, 3] to [3, 2]", async () => {
    const a = tensor([[25.0, -102.0, -1.0], [7.0, -95.0, -38.0]]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.strides).toEqual([3, 1]);
    const c = a.reshape([3, 2]);
    expect(c.shape).toEqual([3, 2]);
    expect(c.strides).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[25.0, -102.0], [-1.0, 7.0], [-95.0, -38.0]]);
});

test("reshape [2, 3] as [3, 2]", async () => {
    const a = tensor([[25.0, -102.0, -1.0], [7.0, -95.0, -38.0]]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.strides).toEqual([3, 1]);
    const c = a.reshapeAs(ones([3, 2]));
    expect(c.shape).toEqual([3, 2]);
    expect(c.strides).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[25.0, -102.0], [-1.0, 7.0], [-95.0, -38.0]]);
});

test("squeeze [] dim ", async () => {
    const a = tensor([]);
    expect(a.shape).toEqual([0]);
    const c = a.squeeze();
    expect(c.shape).toEqual([0]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([]);
});
test("squeeze [] dim 0", async () => {
    const a = tensor([]);
    expect(a.shape).toEqual([0]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([0]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([]);
});
test("squeeze [] dim 1", async () => {
    const a = ones([]);
    expect(() => a.squeeze(1)).toThrow('Dimension out of range (expected to be in range of [-1, 0], but got 1)');
});
test("squeeze [1] dim ", async () => {
    const a = tensor([-220.0]);
    expect(a.shape).toEqual([1]);
    const c = a.squeeze();
    expect(c.shape).toEqual([]);
    expect(c.strides).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(-220.0);
});
test("squeeze [1] dim 0", async () => {
    const a = tensor([-62.0]);
    expect(a.shape).toEqual([1]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([]);
    expect(c.strides).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(-62.0);
});
test("squeeze [1] dim 1", async () => {
    const a = ones([1]);
    expect(() => a.squeeze(1)).toThrow('Dimension out of range (expected to be in range of [-1, 0], but got 1)');
});
test("squeeze [2] dim ", async () => {
    const a = tensor([117.0, 131.0]);
    expect(a.shape).toEqual([2]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([117.0, 131.0]);
});
test("squeeze [2] dim 0", async () => {
    const a = tensor([-164.0, -24.0]);
    expect(a.shape).toEqual([2]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([-164.0, -24.0]);
});
test("squeeze [2] dim 1", async () => {
    const a = ones([2]);
    expect(() => a.squeeze(1)).toThrow('Dimension out of range (expected to be in range of [-1, 0], but got 1)');
});
test("squeeze [1, 1] dim ", async () => {
    const a = tensor([[59.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.squeeze();
    expect(c.shape).toEqual([]);
    expect(c.strides).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(59.0);
});
test("squeeze [1, 1] dim 1", async () => {
    const a = tensor([[-84.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([1]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([-84.0]);
});
test("squeeze [2, 1] dim ", async () => {
    const a = tensor([[-170.0], [62.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([-170.0, 62.0]);
});
test("squeeze [2, 1] dim 1", async () => {
    const a = tensor([[52.0], [-157.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([52.0, -157.0]);
});
test("squeeze [1, 2] dim ", async () => {
    const a = tensor([[97.0, -42.0]]);
    expect(a.shape).toEqual([1, 2]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([97.0, -42.0]);
});
test("squeeze [2, 3, 5] dim ", async () => {
    const a = tensor([[[-71.0, -72.0, -6.0, 49.0, -8.0], [-13.0, 37.0, 132.0, 51.0, 24.0], [-125.0, 56.0, 1.0, -6.0, 31.0]], [[-92.0, 28.0, -110.0, 95.0, 156.0], [-171.0, 18.0, -148.0, -15.0, -166.0], [-90.0, -117.0, 134.0, 139.0, -161.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2, 3, 5]);
    expect(c.strides).toEqual([15, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-71.0, -72.0, -6.0, 49.0, -8.0], [-13.0, 37.0, 132.0, 51.0, 24.0], [-125.0, 56.0, 1.0, -6.0, 31.0]], [[-92.0, 28.0, -110.0, 95.0, 156.0], [-171.0, 18.0, -148.0, -15.0, -166.0], [-90.0, -117.0, 134.0, 139.0, -161.0]]]);
});
test("squeeze [2, 3, 5] dim 1", async () => {
    const a = tensor([[[193.0, 28.0, -84.0, -94.0, 23.0], [141.0, -6.0, -29.0, 30.0, 64.0], [-17.0, 82.0, -154.0, -73.0, 84.0]], [[-50.0, -86.0, -155.0, -77.0, -133.0], [51.0, 171.0, 4.0, 92.0, -24.0], [51.0, -80.0, 27.0, -39.0, -76.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([2, 3, 5]);
    expect(c.strides).toEqual([15, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[193.0, 28.0, -84.0, -94.0, 23.0], [141.0, -6.0, -29.0, 30.0, 64.0], [-17.0, 82.0, -154.0, -73.0, 84.0]], [[-50.0, -86.0, -155.0, -77.0, -133.0], [51.0, 171.0, 4.0, 92.0, -24.0], [51.0, -80.0, 27.0, -39.0, -76.0]]]);
});
test("squeeze [2, 3, 5] dim 3", async () => {
    const a = ones([2, 3, 5]);
    expect(() => a.squeeze(3)).toThrow('Dimension out of range (expected to be in range of [-3, 2], but got 3)');
});
test("squeeze [2, 3, 5] dim -2", async () => {
    const a = tensor([[[40.0, -19.0, -112.0, 167.0, 16.0], [151.0, 17.0, 78.0, 14.0, -90.0], [56.0, 154.0, 171.0, 40.0, -65.0]], [[-168.0, 64.0, -74.0, -144.0, 69.0], [78.0, -153.0, -13.0, -263.0, 108.0], [80.0, -124.0, -9.0, -189.0, -125.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze(-2);
    expect(c.shape).toEqual([2, 3, 5]);
    expect(c.strides).toEqual([15, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[40.0, -19.0, -112.0, 167.0, 16.0], [151.0, 17.0, 78.0, 14.0, -90.0], [56.0, 154.0, 171.0, 40.0, -65.0]], [[-168.0, 64.0, -74.0, -144.0, 69.0], [78.0, -153.0, -13.0, -263.0, 108.0], [80.0, -124.0, -9.0, -189.0, -125.0]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim ", async () => {
    const a = tensor([[[[[-47.0, 76.0]], [[74.0, -63.0]]]], [[[[-69.0, -139.0]], [[56.0, 39.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2, 2, 2]);
    expect(c.strides).toEqual([4, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-47.0, 76.0], [74.0, -63.0]], [[-69.0, -139.0], [56.0, 39.0]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim 0", async () => {
    const a = tensor([[[[[-15.0, -52.0]], [[23.0, 104.0]]]], [[[[-56.0, -194.0]], [[34.0, 91.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([2, 1, 2, 1, 2]);
    expect(c.strides).toEqual([4, 4, 2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[-15.0, -52.0]], [[23.0, 104.0]]]], [[[[-56.0, -194.0]], [[34.0, 91.0]]]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim 1", async () => {
    const a = tensor([[[[[0.0, 71.0]], [[86.0, 18.0]]]], [[[[-145.0, -26.0]], [[161.0, -40.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([2, 2, 1, 2]);
    expect(c.strides).toEqual([4, 2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[0.0, 71.0]], [[86.0, 18.0]]], [[[-145.0, -26.0]], [[161.0, -40.0]]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim (1, 2, 3)", async () => {
    const a = tensor([[[[[57.0, 33.0]], [[64.0, 69.0]]]], [[[[113.0, 122.0]], [[83.0, -97.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze([1, 2, 3]);
    expect(c.shape).toEqual([2, 2, 2]);
    expect(c.strides).toEqual([4, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[57.0, 33.0], [64.0, 69.0]], [[113.0, 122.0], [83.0, -97.0]]]);
});

test("unsqueeze [] dim 0", async () => {
    const a = tensor([]);
    expect(a.shape).toEqual([0]);
    const c = a.unsqueeze(0);
    expect(c.shape).toEqual([1, 0]);
    expect(c.strides).toEqual([0, 1]);
    expect(await c.toArrayAsync()).toEqual([[]]);
});
test("unsqueeze [] dim 1", async () => {
    const a = tensor([]);
    expect(a.shape).toEqual([0]);
    const c = a.unsqueeze(1);
    expect(c.shape).toEqual([0, 1]);
    expect(c.strides).toEqual([1, 1]);
    expect(await c.toArrayAsync()).toEqual([]);
});
test("unsqueeze [] dim -2", async () => {
    const a = tensor([]);
    expect(a.shape).toEqual([0]);
    const c = a.unsqueeze(-2);
    expect(c.shape).toEqual([1, 0]);
    expect(c.strides).toEqual([0, 1]);
    expect(await c.toArrayAsync()).toEqual([[]]);
});
test("unsqueeze [] dim -3", async () => {
    const a = ones([]);
    expect(() => a.unsqueeze(-3)).toThrow('Dimension out of range (expected to be in range of [-2, 1], but got -3)');
});
test("unsqueeze [1] dim 0", async () => {
    const a = tensor([-38.0]);
    expect(a.shape).toEqual([1]);
    const c = a.unsqueeze(0);
    expect(c.shape).toEqual([1, 1]);
    expect(c.strides).toEqual([1, 1]);
    expect(await c.toArrayAsync()).toEqual([[-38.0]]);
});
test("unsqueeze [1] dim 1", async () => {
    const a = tensor([-28.0]);
    expect(a.shape).toEqual([1]);
    const c = a.unsqueeze(1);
    expect(c.shape).toEqual([1, 1]);
    expect(c.strides).toEqual([1, 1]);
    expect(await c.toArrayAsync()).toEqual([[-28.0]]);
});
test("unsqueeze [2] dim 0", async () => {
    const a = tensor([23.0, -53.0]);
    expect(a.shape).toEqual([2]);
    const c = a.unsqueeze(0);
    expect(c.shape).toEqual([1, 2]);
    expect(c.strides).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[23.0, -53.0]]);
});
test("unsqueeze [2] dim 1", async () => {
    const a = tensor([-11.0, -33.0]);
    expect(a.shape).toEqual([2]);
    const c = a.unsqueeze(1);
    expect(c.shape).toEqual([2, 1]);
    expect(c.strides).toEqual([1, 1]);
    expect(await c.toArrayAsync()).toEqual([[-11.0], [-33.0]]);
});
test("unsqueeze [2] dim -2", async () => {
    const a = tensor([113.0, 99.0]);
    expect(a.shape).toEqual([2]);
    const c = a.unsqueeze(-2);
    expect(c.shape).toEqual([1, 2]);
    expect(c.strides).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[113.0, 99.0]]);
});
test("unsqueeze [2] dim -3", async () => {
    const a = ones([2]);
    expect(() => a.unsqueeze(-3)).toThrow('Dimension out of range (expected to be in range of [-2, 1], but got -3)');
});
test("unsqueeze [1, 1] dim 0", async () => {
    const a = tensor([[-2.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.unsqueeze(0);
    expect(c.shape).toEqual([1, 1, 1]);
    expect(c.strides).toEqual([1, 1, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-2.0]]]);
});
test("unsqueeze [1, 1] dim 1", async () => {
    const a = tensor([[-133.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.unsqueeze(1);
    expect(c.shape).toEqual([1, 1, 1]);
    expect(c.strides).toEqual([1, 1, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-133.0]]]);
});
test("unsqueeze [2, 1]", async () => {
    const a = tensor([[-116.0], [-163.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.unsqueeze();
    expect(c.shape).toEqual([1, 2, 1]);
    expect(c.strides).toEqual([2, 1, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-116.0], [-163.0]]]);
});
test("unsqueeze [2, 1] dim 0", async () => {
    const a = tensor([[-116.0], [-163.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.unsqueeze(0);
    expect(c.shape).toEqual([1, 2, 1]);
    expect(c.strides).toEqual([2, 1, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-116.0], [-163.0]]]);
});
test("unsqueeze [2, 1] dim 1", async () => {
    const a = tensor([[-117.0], [-104.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.unsqueeze(1);
    expect(c.shape).toEqual([2, 1, 1]);
    expect(c.strides).toEqual([1, 1, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-117.0]], [[-104.0]]]);
});
test("unsqueeze [1, 2] dim 0", async () => {
    const a = tensor([[-36.0, 169.0]]);
    expect(a.shape).toEqual([1, 2]);
    const c = a.unsqueeze(0);
    expect(c.shape).toEqual([1, 1, 2]);
    expect(c.strides).toEqual([2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-36.0, 169.0]]]);
});
test("unsqueeze [1, 2] dim 1", async () => {
    const a = tensor([[116.0, -113.0]]);
    expect(a.shape).toEqual([1, 2]);
    const c = a.unsqueeze(1);
    expect(c.shape).toEqual([1, 1, 2]);
    expect(c.strides).toEqual([2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[116.0, -113.0]]]);
});
test("unsqueeze [1, 2] dim -4", async () => {
    const a = ones([1, 2]);
    expect(() => a.unsqueeze(-4)).toThrow('Dimension out of range (expected to be in range of [-3, 2], but got -4)');
});
test("unsqueeze [2, 3, 5] dim 0", async () => {
    const a = tensor([[[130.0, -25.0, -68.0, -70.0, -72.0], [-22.0, -167.0, 14.0, 9.0, -148.0], [12.0, -54.0, -48.0, -70.0, -85.0]], [[-88.0, 9.0, 151.0, -38.0, 24.0], [-93.0, 52.0, -226.0, 82.0, -4.0], [39.0, -97.0, -51.0, 84.0, -10.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.unsqueeze(0);
    expect(c.shape).toEqual([1, 2, 3, 5]);
    expect(c.strides).toEqual([30, 15, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[130.0, -25.0, -68.0, -70.0, -72.0], [-22.0, -167.0, 14.0, 9.0, -148.0], [12.0, -54.0, -48.0, -70.0, -85.0]], [[-88.0, 9.0, 151.0, -38.0, 24.0], [-93.0, 52.0, -226.0, 82.0, -4.0], [39.0, -97.0, -51.0, 84.0, -10.0]]]]);
});
test("unsqueeze [2, 3, 5] dim 3", async () => {
    const a = tensor([[[-92.0, 48.0, -18.0, -133.0, -36.0], [-36.0, -30.0, -85.0, 112.0, 76.0], [27.0, 2.0, -12.0, 67.0, -187.0]], [[-23.0, 64.0, 254.0, 88.0, 73.0], [-198.0, -90.0, 144.0, 66.0, 131.0], [-155.0, 103.0, 22.0, 41.0, -1.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.unsqueeze(3);
    expect(c.shape).toEqual([2, 3, 5, 1]);
    expect(c.strides).toEqual([15, 5, 1, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[-92.0], [48.0], [-18.0], [-133.0], [-36.0]], [[-36.0], [-30.0], [-85.0], [112.0], [76.0]], [[27.0], [2.0], [-12.0], [67.0], [-187.0]]], [[[-23.0], [64.0], [254.0], [88.0], [73.0]], [[-198.0], [-90.0], [144.0], [66.0], [131.0]], [[-155.0], [103.0], [22.0], [41.0], [-1.0]]]]);
});
test("unsqueeze [2, 3, 5] dim -2", async () => {
    const a = tensor([[[-40.0, 4.0, -87.0, -51.0, -27.0], [136.0, 169.0, -33.0, 107.0, -56.0], [255.0, -123.0, -58.0, 47.0, 116.0]], [[-12.0, 207.0, -30.0, -138.0, -110.0], [-132.0, -48.0, -157.0, -164.0, -41.0], [-4.0, 31.0, 177.0, 23.0, 1.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.unsqueeze(-2);
    expect(c.shape).toEqual([2, 3, 1, 5]);
    expect(c.strides).toEqual([15, 5, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[-40.0, 4.0, -87.0, -51.0, -27.0]], [[136.0, 169.0, -33.0, 107.0, -56.0]], [[255.0, -123.0, -58.0, 47.0, 116.0]]], [[[-12.0, 207.0, -30.0, -138.0, -110.0]], [[-132.0, -48.0, -157.0, -164.0, -41.0]], [[-4.0, 31.0, 177.0, 23.0, 1.0]]]]);
});
test("unsqueeze [2, 3, 5] dim -5", async () => {
    const a = ones([2, 3, 5]);
    expect(() => a.unsqueeze(-5)).toThrow('Dimension out of range (expected to be in range of [-4, 3], but got -5)');
});
test("unsqueeze [2, 1, 2, 1, 2] dim 0", async () => {
    const a = tensor([[[[[-33.0, -202.0]], [[-18.0, 143.0]]]], [[[[-92.0, 0.0]], [[-25.0, 40.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.unsqueeze(0);
    expect(c.shape).toEqual([1, 2, 1, 2, 1, 2]);
    expect(c.strides).toEqual([8, 4, 4, 2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[[-33.0, -202.0]], [[-18.0, 143.0]]]], [[[[-92.0, 0.0]], [[-25.0, 40.0]]]]]]);
});
test("unsqueeze [2, 1, 2, 1, 2] dim 1", async () => {
    const a = tensor([[[[[59.0, 89.0]], [[-48.0, -94.0]]]], [[[[-12.0, 132.0]], [[127.0, 3.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.unsqueeze(1);
    expect(c.shape).toEqual([2, 1, 1, 2, 1, 2]);
    expect(c.strides).toEqual([4, 4, 4, 2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[[59.0, 89.0]], [[-48.0, -94.0]]]]], [[[[[-12.0, 132.0]], [[127.0, 3.0]]]]]]);
});

// test("view [] to []", async () => {
//     const a = ones([]);
//     expect(() => a.view([])).toThrow("shape '[]' is invalid for input of size 0");
// });
// test("view [] to [1]", async () => {
//     const a = ones([]);
//     expect(() => a.view([1])).toThrow("shape '[1]' is invalid for input of size 0");
// });
test("view [] to [1, 0]", async () => {
    const a = tensor([]);
    expect(a.shape).toEqual([0]);
    const c = a.view([1, 0]);
    expect(c.shape).toEqual([1, 0]);
    expect(c.strides).toEqual([1, 1]);
    expect(await c.toArrayAsync()).toEqual([[]]);
});
// test("view [] to [1, 1]", async () => {
//     const a = ones([]);
//     expect(() => a.view([1, 1])).toThrow("shape '[1, 1]' is invalid for input of size 0");
// });
test("view [1] to []", async () => {
    const a = tensor([34.0]);
    expect(a.shape).toEqual([1]);
    const c = a.view([]);
    expect(c.shape).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(34.0);
});
test("view [2] to []", async () => {
    const a = ones([2]);
    expect(() => a.view([])).toThrow("shape '[]' is invalid for input of size 2");
});
test("view [1] to [1]", async () => {
    const a = tensor([-116.0]);
    expect(a.shape).toEqual([1]);
    const c = a.view([1]);
    expect(c.shape).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([-116.0]);
});
test("view [2] to [-1]", async () => {
    const a = tensor([-50.0, 82.0]);
    expect(a.shape).toEqual([2]);
    const c = a.view([-1]);
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([-50.0, 82.0]);
});
test("view [1, 1] to []", async () => {
    const a = tensor([[120.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.view([]);
    expect(c.shape).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(120.0);
});
test("view [2, 3] to [2, -1]", async () => {
    const a = tensor([[112.0, -103.0, -218.0], [-80.0, -32.0, 237.0]]);
    expect(a.shape).toEqual([2, 3]);
    const c = a.view([2, -1]);
    expect(c.shape).toEqual([2, 3]);
    expect(await c.toArrayAsync()).toEqual([[112.0, -103.0, -218.0], [-80.0, -32.0, 237.0]]);
});
test("view [2, 3] to [-1, 3]", async () => {
    const a = tensor([[-138.0, 103.0, 58.0], [-182.0, 64.0, 8.0]]);
    expect(a.shape).toEqual([2, 3]);
    const c = a.view([-1, 3]);
    expect(c.shape).toEqual([2, 3]);
    expect(await c.toArrayAsync()).toEqual([[-138.0, 103.0, 58.0], [-182.0, 64.0, 8.0]]);
});
test("view [2, 3] to [3, 2]", async () => {
    const a = tensor([[25.0, -102.0, -1.0], [7.0, -95.0, -38.0]]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.strides).toEqual([3, 1]);
    const c = a.view([3, 2]);
    expect(c.shape).toEqual([3, 2]);
    expect(c.strides).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[25.0, -102.0], [-1.0, 7.0], [-95.0, -38.0]]);
});
test("view [2, 15] to [3, -1]", async () => {
    const a = tensor([[120.0, -202.0, 12.0, -39.0, 81.0, 92.0, 247.0, -497.0, -37.0, -130.0, 126.0, -178.0, 20.0, 32.0, 118.0], [4.0, -92.0, -18.0, 115.0, 33.0, 94.0, -136.0, -5.0, 18.0, 96.0, 29.0, 146.0, -42.0, 20.0, 109.0]]);
    expect(a.shape).toEqual([2, 15]);
    expect(a.strides).toEqual([15, 1]);
    const c = a.view([3, -1]);
    expect(c.shape).toEqual([3, 10]);
    expect(c.strides).toEqual([10, 1]);
    expect(await c.toArrayAsync()).toEqual([[120.0, -202.0, 12.0, -39.0, 81.0, 92.0, 247.0, -497.0, -37.0, -130.0], [126.0, -178.0, 20.0, 32.0, 118.0, 4.0, -92.0, -18.0, 115.0, 33.0], [94.0, -136.0, -5.0, 18.0, 96.0, 29.0, 146.0, -42.0, 20.0, 109.0]]);
});
test("view [2, 15] to [4, -1]", async () => {
    const a = ones([2, 15]);
    expect(() => a.view([4, -1])).toThrow("shape '[4,-1]' is invalid for input of size 30");
});
test("view [2, 15] to [1, 15, 2]", async () => {
    const a = tensor([[-77.0, 14.0, 17.0, 84.0, -21.0, 54.0, -97.0, 126.0, -63.0, 129.0, 17.0, 46.0, 16.0, 215.0, -83.0], [144.0, -88.0, 114.0, 130.0, 34.0, -32.0, 92.0, -1.0, -60.0, -59.0, 10.0, 170.0, -137.0, 4.0, -138.0]]);
    expect(a.shape).toEqual([2, 15]);
    expect(a.strides).toEqual([15, 1]);
    const c = a.view([1, 15, 2]);
    expect(c.shape).toEqual([1, 15, 2]);
    expect(c.strides).toEqual([30, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-77.0, 14.0], [17.0, 84.0], [-21.0, 54.0], [-97.0, 126.0], [-63.0, 129.0], [17.0, 46.0], [16.0, 215.0], [-83.0, 144.0], [-88.0, 114.0], [130.0, 34.0], [-32.0, 92.0], [-1.0, -60.0], [-59.0, 10.0], [170.0, -137.0], [4.0, -138.0]]]);
});
test("view [2, 3] to [1, 1, 3, 1, 2, 1, 1]", async () => {
    const a = tensor([[135.0, -38.0, 56.0], [28.0, 17.0, 96.0]]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.strides).toEqual([3, 1]);
    const c = a.view([1, 1, 3, 1, 2, 1, 1]);
    expect(c.shape).toEqual([1, 1, 3, 1, 2, 1, 1]);
    expect(c.strides).toEqual([6, 6, 2, 2, 1, 1, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[[[135.0]], [[-38.0]]]], [[[[56.0]], [[28.0]]]], [[[[17.0]], [[96.0]]]]]]]);
});
test("view [2, 3] as [3, 2]", async () => {
    const a = tensor([[25.0, -102.0, -1.0], [7.0, -95.0, -38.0]]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.strides).toEqual([3, 1]);
    const c = a.viewAs(ones([3, 2]));
    expect(c.shape).toEqual([3, 2]);
    expect(c.strides).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[25.0, -102.0], [-1.0, 7.0], [-95.0, -38.0]]);
});
