import { tensor } from "./ops_artisanal";

test("array tensors are source nodes", () => {
    const t = tensor([1, 2, 3]);
    expect(t.node.isSource).toBe(true);
});
