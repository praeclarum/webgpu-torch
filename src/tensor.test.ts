import { UntypedStorage } from "./storage";
import { Tensor } from "./index";

test("create tensor with storage and dtype", () => {
    const storage = new UntypedStorage();
    const tensor = new Tensor(storage, "float32");
    expect(tensor).toBeInstanceOf(Tensor);
    expect(tensor.untypedStorage).toBeInstanceOf(UntypedStorage);
});
