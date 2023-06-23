import { ones } from "./factories";
import { tensor } from "./ops_artisanal";

test("sigmoid(1).size == 307", async () => {
    // prime number > 300
    const n = 307;
    const x = ones(n);
    const y = x.sigmoid();
    expect(y.shape[0]).toBe(n);
    const a = await y.toArrayAsync();
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
    expect(a).toEqual([6]);
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
    const a = await y.toArrayAsync();
    expect(a[0]).toBeCloseTo(2);
});

test("mean vector grad", async () => {
    const x = tensor({data:[1, 2, 3, 4], requiresGrad: true});
    const y = x.mean();
    y.backward();
    const a = await x.grad!.toArrayAsync();
    expect(a).toEqual([0.25, 0.25, 0.25, 0.25]);
});
