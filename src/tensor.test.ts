import { UntypedStorage } from "./storage";
import { Tensor } from "./index";

test("create tensor with storage and dtype", () => {
    const storage = new UntypedStorage();
    const tensor = new Tensor(storage, "float32");
    expect(tensor).toBeInstanceOf(Tensor);
    expect(tensor.untypedStorage).toBeInstanceOf(UntypedStorage);
    expect(tensor.dtype).toBe("float32");
});

test("can toggle requiresGrad", () => {
    const storage = new UntypedStorage();
    const tensor = new Tensor(storage, "float32");
    expect(tensor.requiresGrad).toBe(false);
    tensor.requiresGrad = true;
    expect(tensor.requiresGrad).toBe(true);
});
