import { ComputedNode } from "./graph";
import { tensor } from "./ops_artisanal";
import { Tensor } from "./tensor";

test("array tensors are source nodes", async () => {
    const x = tensor([1, 2, 3]);
    expect(x.node.isSource).toBe(true);
    expect(await x.node.storage.toArrayAsync(x.dtype)).toEqual([1, 2, 3]);
});

test("unop tensors are computed nodes", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.neg();
    expect(y.node.isComputed).toBe(true);
    expect(await y.node.storage.toArrayAsync(y.dtype)).toEqual([-1, -2, -3]);
    const node = y.node as ComputedNode;
    expect(node.inputs).toEqual([x.node]);
});

test("deep equal-width graphs use minimal buffers", async () => {
    const x = tensor([1, 2, 3]);
    let y = x;
    const ys: Tensor[] = [];
    const depth = 101;
    for (let i = 0; i < depth; i++) {
        y = y.neg();
        ys.push(y);
    }
    expect(y.node.isComputed).toBe(true);
    expect(await y.node.storage.toArrayAsync(y.dtype)).toEqual([-1, -2, -3]);
    const node = y.node as ComputedNode;
    expect(node.inputs).toEqual([ys[depth - 2].node]);
});
