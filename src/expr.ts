export type ExprCode = number | string;

export type ExprNode = string | number | [string, ExprNode[]];

export type ParsedExpr = ExprNode;

export type EvalEnv = { [name: string]: any };

function lexn(code: string): (string | number)[] {
    const tokens: (string | number)[] = [];
    const n = code.length;
    let i = 0;
    while (i < n) {
        const c = code[i];
        if (c === " " || c === "\t" || c === "\n") {
            i++;
            continue;
        }
        if (c === "+" || c === "-" || c === "*" || c === "/") {
            tokens.push(c);
            i++;
            continue;
        }
        if (c === "(" || c === ")") {
            tokens.push(c);
            i++;
            continue;
        }
        if (c >= "0" && c <= "9") {
            let s = "";
            while (i < n && (code[i] >= "0" && code[i] <= "9" || code[i] === ".")) {
                s += code[i];
                i++;
            }
            tokens.push(parseFloat(s));
            continue;
        }
        if (
            (c >= "a" && c <= "z") ||
            (c >= "A" && c <= "Z") ||
            c === "_"
        ) {
            let s = "";
            while (
                i < n &&
                ((code[i] >= "a" && code[i] <= "z") ||
                    (code[i] >= "A" && code[i] <= "Z") ||
                    (code[i] >= "0" && code[i] <= "9") ||
                    code[i] === "_")
            ) {
                s += code[i];
                i++;
            }
            tokens.push(s);
            continue;
        }
        throw new Error(`Unexpected character ${c}`);
    }
    return tokens;
}

export function parseExpr(code: string): ExprNode {
    const tokens = lexn(code);
    const expr = parseExpr(0);
    if (expr === null) {
        throw new Error("Missing expression");
    }
    if (expr[1] < tokens.length) {
        console.log("bad tree", expr[0], expr[1], tokens.length, tokens[expr[1]])
        throw new Error(`Unexpected token: '${tokens[expr[1]]}' after parsing: ${JSON.stringify(expr[0])}`);
    }
    return expr[0];
    function parsePrimary(i: number): [ExprNode, number]|null {
        if (i >= tokens.length) {
            return null;
        }
        const t = tokens[i];
        if (typeof t === "number") {
            return [t, i + 1];
        }
        if (typeof t === "string") {
            if (t === "(") {
                const expr = parseExpr(i + 1);
                if (expr === null) {
                    throw new Error("Missing expression");
                }
                i = expr[1];
                if (i >= tokens.length) {
                    throw new Error("Unexpected end of expression");
                }
                const t2 = tokens[i];
                if (typeof t2 !== "string" || t2 !== ")") {
                    throw new Error("Expected )");
                }
                return [expr[0], i + 1];
            }
            return [t, i + 1];
        }
        return null;
    }
    function parseExpr(i: number): [ExprNode, number]|null {
        return parseAddOrSubtract(i);
    }
    function parseAddOrSubtract(i: number): [ExprNode, number]|null {
        let expr = parseMultiplyOrDivide(i);
        if (expr === null) {
            return null;
        }
        i = expr[1];
        while (i < tokens.length) {
            const t = tokens[i];
            if (typeof t !== "string") {
                break;
            }
            if (t === "+" || t === "-") {
                const expr2 = parseMultiplyOrDivide(i + 1);
                if (expr2 === null) {
                    throw new Error("Missing expression");
                }
                expr = [[t, [expr[0], expr2[0]]], expr2[1]];
                i = expr2[1];
                continue;
            }
            break;
        }
        return expr;
    }
    function parseMultiplyOrDivide(i: number): [ExprNode, number]|null {
        let expr = parsePrimary(i);
        if (expr === null) {
            return null;
        }
        i = expr[1];
        while (i < tokens.length) {
            const t = tokens[i];
            if (typeof t !== "string") {
                break;
            }
            if (t === "*" || t === "/") {
                const expr2 = parsePrimary(i + 1);
                if (expr2 === null) {
                    throw new Error("Missing expression");
                }
                expr = [[t, [expr[0], expr2[0]]], expr2[1]];
                i = expr2[1];
                continue;
            }
            break;
        }
        return expr;
    }
}

export function substituteIdentifiers(ast: ExprNode, subs: { [name: string]: ExprNode }): ExprNode {
    if (typeof ast === "number") {
        return ast;
    }
    if (typeof ast === "string") {
        if (ast in subs) {
            return subs[ast];
        }
        return ast;
    }
    const newChildren: ExprNode[] = [];
    for (const child of ast[1]) {
        newChildren.push(substituteIdentifiers(child, subs));
    }
    return [ast[0], newChildren];
}

export function exprNodeToString(ast: ExprNode): string {
    if (typeof ast === "number") {
        return ast.toString();
    }
    if (typeof ast === "string") {
        return ast;
    }
    if (ast[0] === "+") {
        return `(${exprNodeToString(ast[1][0])} + ${exprNodeToString(ast[1][1])})`;
    }
    if (ast[0] === "-") {
        return `(${exprNodeToString(ast[1][0])} - ${exprNodeToString(ast[1][1])})`;
    }
    if (ast[0] === "*") {
        return `(${exprNodeToString(ast[1][0])} * ${exprNodeToString(ast[1][1])})`;
    }
    if (ast[0] === "/") {
        return `(${exprNodeToString(ast[1][0])} / ${exprNodeToString(ast[1][1])})`;
    }
    throw new Error("Unknown AST node type");
}


/*== COMPILER ==*/

const OP_CONSTANT = 0;
const OP_READ = 1;
const OP_ADD = 2;
const OP_SUBTRACT = 3;
const OP_MULTIPLY = 4;
const OP_DIVIDE = 5;

export type CompiledExpr = (env: { [name: string]: any }) => number;

export function compileCode(code: ExprCode): CompiledExpr {
    if (typeof code === "number") {
        return () => code;
    }
    const expr = parseExpr(code);
    // console.log("parsed", expr);
    const instructions: [number, (number|string|null)][] = [];
    function emit(node: ExprNode) {
        if (typeof node === "number") {
            instructions.push([OP_CONSTANT, node]);
            return;
        }
        if (typeof node === "string") {
            instructions.push([OP_READ, node]);
            return;
        }
        if (node[0] === "+") {
            emit(node[1][0]);
            emit(node[1][1]);
            instructions.push([OP_ADD, null]);
            return;
        }
        if (node[0] === "-") {
            emit(node[1][0]);
            emit(node[1][1]);
            instructions.push([OP_SUBTRACT, null]);
            return;
        }
        if (node[0] === "*") {
            emit(node[1][0]);
            emit(node[1][1]);
            instructions.push([OP_MULTIPLY, null]);
            return;
        }
        if (node[0] === "/") {
            emit(node[1][0]);
            emit(node[1][1]);
            instructions.push([OP_DIVIDE, null]);
            return;
        }
        throw new Error(`Unknown node type ${node}`);
    }
    emit(expr);
    // console.log("ops", ops);
    return (env: { [name: string]: any }) => {
        // Evaluate the ops
        const stack: number[] = [];
        for (const ins of instructions) {
            const op = ins[0];
            if (op === 0) {
                stack.push(ins[1] as number);
            }
            else if (op == 1) {
                stack.push(env[ins[1] as string]);
            }
            else {
                const b = stack.pop() as number;
                const a = stack.pop() as number;
                switch (op) {
                    case 2:
                        stack.push(a + b);
                        break;
                    case 3:
                        stack.push(a - b);
                        break;
                    case 4:
                        stack.push(a * b);
                        break;
                    case 5:
                        stack.push(a / b);
                        break;
                }
            }
        }
        return stack[0] as number;
    };
}

export function evalCode(input: ExprCode, env: { [name: string]: any }): number {
    if (typeof input === "number") {
        return input;
    }
    const c = compileCode(input);
    return c(env);
}
