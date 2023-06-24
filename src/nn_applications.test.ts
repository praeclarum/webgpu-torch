import { tensor } from "./ops_artisanal";
import { DeepSDF } from "./nn_applications";

test("DeepSDF outputs batched distances", async () => {
    const model = new DeepSDF();
    const batchedPoints = tensor([[0, 0, 0], [1, 1, 1]]);
    // const batchedDistances = model.forward(batchedPoints);
    // expect(batchedDistances.shape).toEqual([2, 1]);
});
