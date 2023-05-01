import { getDevice } from "./devices";

test("getDevice with cpu", () => {
    const d = getDevice("cpu");
    expect(d.id).toBe("cpu");
    expect(d.type).toBe("cpu");
});
