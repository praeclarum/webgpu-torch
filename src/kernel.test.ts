import { tensor } from "./ops_artisanal";

test("cpu abs value with grad", async () => {
    const x = tensor({data:[[-1, 2, -3], [4, -5, 6]], requiresGrad:true, device: "cpu"});
    expect(x.device.type).toBe("cpu");
    const y = x.abs();
    expect(await y.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(y.requiresGrad).toBe(true);
});
