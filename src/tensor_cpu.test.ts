import { TensorCPU } from "./tensor_cpu";

import { getDevice } from "./devices";

const cpu = getDevice("cpu");

test("1D 1 element shape", () => {
    const t = cpu.tensor([1], "float32");
    expect(t.shape).toEqual([1]);
});

test("2D element shape", () => {
    const t = cpu.tensor(
        [
            [2, 3, 4],
            [5, 6, 7],
        ],
        "float32"
    );
    expect(t.shape).toEqual([2, 3]);
});

test("2D element data", () => {
    const t = cpu.tensor(
        [
            [2, 3, 4],
            [5, 6, 7],
        ],
        "float32"
    ) as TensorCPU;
    expect(t.data).not.toBeNull();
    expect(t.data).toEqual(new Float32Array([2, 3, 4, 5, 6, 7]));
});

test("2D transpose", () => {
    const t = cpu.tensor(
        [
            [2, 3, 4],
            [5, 6, 7],
        ],
        "float32"
    );
    const tr = t.t();
    expect(t.shape).toEqual([2, 3]);
    expect(t.get(0, 1)).toEqual(3);
    expect(tr.shape).toEqual([3, 2]);
    expect(tr.get(0, 1)).toEqual(5);
});
