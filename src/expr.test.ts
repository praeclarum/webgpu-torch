import { compileExpr, evalExpr } from "./expr";

test("just a number", () => {
    const expr = "3.14";
    expect(evalExpr(expr, {})).toEqual(3.14);
});

test("add expression", () => {
    const expr = "2+3";
    expect(evalExpr(expr, {})).toEqual(5);
});

test("subtract expression", () => {
    const expr = "5-2";
    expect(evalExpr(expr, {})).toEqual(3);
});

test("divide expression", () => {
    const expr = "6/2";
    expect(evalExpr(expr, {})).toEqual(3);
});

test("multiadd expression", () => {
    const expr = "2+3+4";
    expect(evalExpr(expr, {})).toEqual(9);
});

test("parens expression", () => {
    const expr = "(2+3)*4";
    expect(evalExpr(expr, {})).toEqual(20);
});

test("name lookup", () => {
    const expr = "x";
    expect(evalExpr(expr, { x: 3 })).toEqual(3);
});

test("compile number", () => {
    const expr = 3;
    const compiled = compileExpr(expr);
    expect(compiled({})).toEqual(3);
});
