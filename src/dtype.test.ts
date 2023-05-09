import { dtypeByteSize, getDtype } from "./dtype";

test("dtype byte sizes", () => {
    expect(dtypeByteSize("uint8")).toBe(1);
    expect(dtypeByteSize("int32")).toBe(4);
    expect(dtypeByteSize("uint32")).toBe(4);
    expect(dtypeByteSize("float32")).toBe(4);
});

test("bad byte size throws", () => {
    expect(() => dtypeByteSize("bad" as any)).toThrow();
});

test("getDtype works for dtypes", () => {
    expect(getDtype("uint8")).toBe("uint8");
    expect(getDtype("int32")).toBe("int32");
    expect(getDtype("uint32")).toBe("uint32");
    expect(getDtype("float32")).toBe("float32");
});

test("getDtype works for typed arrays", () => {
    expect(getDtype(new Uint8Array(1))).toBe("uint8");
    expect(getDtype(new Int32Array(1))).toBe("int32");
    expect(getDtype(new Uint32Array(1))).toBe("uint32");
    expect(getDtype(new Float32Array(1))).toBe("float32");
});

test("bad getDtype throws", () => {
    expect(() => getDtype("bad" as any)).toThrow();
});
