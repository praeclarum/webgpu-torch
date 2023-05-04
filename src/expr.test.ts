import { compileCode, evalCode, parseCode } from "./expr";

test("just a number", () => {
    const expr = "3.14";
    expect(evalCode(expr, {})).toEqual(3.14);
});

test("add expression", () => {
    const expr = "2+3";
    expect(evalCode(expr, {})).toEqual(5);
});

test("subtract expression", () => {
    const expr = "5-2";
    expect(evalCode(expr, {})).toEqual(3);
});

test("divide expression", () => {
    const expr = "6/2";
    expect(evalCode(expr, {})).toEqual(3);
});

test("multiadd expression", () => {
    const expr = "2+3+4";
    expect(evalCode(expr, {})).toEqual(9);
});

test("parens expression", () => {
    const expr = "(2+3)*4";
    expect(evalCode(expr, {})).toEqual(20);
});

test("name lookup", () => {
    const expr = "x";
    expect(evalCode(expr, { x: 3 })).toEqual(3);
});

test("compile number", () => {
    const expr = 3;
    const compiled = compileCode(expr);
    expect(compiled({})).toEqual(3);
});

test("parse apply", () => {
    const expr = "f(3)";
    const parsed = parseCode(expr);
    expect(parsed).toEqual(["apply", ["f", 3]]);
});

test("parse assign", () => {
    const expr = "x = 3";
    const parsed = parseCode(expr);
    expect(parsed).toEqual(["assign", ["x", 3]]);
});

test("parse statements", () => {
    const expr = "x = 3; y = 4; x+y";
    const parsed = parseCode(expr);
    expect(parsed).toEqual(["statements", [["assign", ["x", 3]], ["assign", ["y", 4]], ["+", ["x", "y"]]]]);
});
