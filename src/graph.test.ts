import { ComputedNode } from "./graph";
import { tensor } from "./ops_artisanal";
import { Tensor } from "./tensor";

test("array tensors are source nodes", async () => {
    const x = tensor([1, 2, 3]);
    expect(x.node.node.isSource).toBe(true);
    expect(await x.node.node.storages[0].toArrayAsync(x.dtype)).toEqual([1, 2, 3]);
});

test("unop tensors are computed nodes", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.neg();
    expect(y.node.node.isComputed).toBe(true);
    expect(await y.node.node.storages[0].toArrayAsync(y.dtype)).toEqual([-1, -2, -3]);
    const node = y.node.node as ComputedNode;
    expect(node.inputs).toEqual([x.node]);
});

test("deep equal-width graphs use minimal buffers", async () => {
    const x = tensor([1, 2, 3]);
    let y = x;
    const ys: Tensor[] = [];
    const depth = 51;
    for (let i = 0; i < depth; i++) {
        y = y.neg();
        ys.push(y);
    }
    expect(y.node.node.isComputed).toBe(true);
    expect(await y.node.node.storages[0].toArrayAsync(y.dtype)).toEqual([-1, -2, -3]);
    const node = y.node.node as ComputedNode;
    expect(node.inputs).toEqual([ys[depth - 2].node]);
});

test("inplace ops handled correctly", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.neg();
    x.neg_();
    const z = x.neg();
    expect(await x.toArrayAsync()).toEqual([-1, -2, -3]);
    expect(await y.toArrayAsync()).toEqual([-1, -2, -3]);
    expect(await z.toArrayAsync()).toEqual([1, 2, 3]);
});

test("temporaries are not retained", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.neg();
    const z = y.neg();
    expect(await z.toArrayAsync()).toEqual([1, 2, 3]);
    expect(y.node.node.storageAvailable).toEqual(false);
    expect(z.node.node.storageAvailable).toEqual(true);
});

test("aliases are retained", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.neg();
    const z = y.neg();
    const w = y.neg();
    expect(await z.toArrayAsync()).toEqual([1, 2, 3]);
    expect(await w.toArrayAsync()).toEqual([1, 2, 3]);
    expect(y.node.node.storageAvailable).toEqual(true);
    expect(z.node.node.storageAvailable).toEqual(true);
    expect(w.node.node.storageAvailable).toEqual(true);
});
