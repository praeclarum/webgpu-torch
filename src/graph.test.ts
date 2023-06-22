import { tensor } from "./ops_artisanal";

test("array tensors are source nodes", () => {
    const x = tensor([1, 2, 3]);
    expect(x.node.isSource).toBe(true);
});

test("unop tensors are computed nodes", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.neg();
    expect(y.node.isComputed).toBe(true);
    expect(await y.node.storage.toArrayAsync(y.dtype)).toEqual([-1, -2, -3]);
});
