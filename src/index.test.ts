import { testExport } from "./index";

test("exports work", () => {
    expect(testExport(42)).toBe(42);
});
