import { ones } from "./factories";

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
