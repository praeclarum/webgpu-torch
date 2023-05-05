export type ExprCode = number | string;

export type ExprNodeType = "apply" | "block" | "assign" | "if" | "return" | "statements" | "+" | "-" | "*" | "/" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||" | "!" | "~" | "^" | "%";

export type ExprNode = string | number | [ExprNodeType, ExprNode[]];

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
        if (c === "+" || c === "-" || c === "*" || c === "/" || c == "," || c == ";" || c == ":" || c == "?" || c == "^" || c == "%" || c == "~" || c == "(" || c == ")" || c == "[" || c == "]" || c == "{" || c == "}" || c == ".") {
            tokens.push(c);
            i++;
            continue;
        }
        if (c == "=" || c == "<" || c == ">" || c == "!") {
            if (i + 1 < n && code[i + 1] == "=") {
                tokens.push(c + "=");
                i += 2;
                continue;
            }
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

function tokenIsIdent(token: string): boolean {
    if (token.length <= 0)
        return false;
    const c = token[0];
    if (!((c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || c === "_")) {
        return false;
    }
    if (c === "if" || c === "else" || c == "return") {
        return false;
    }
    return true;
}

type ParseState = [ExprNode, number];

export function parseCode(code: ExprCode): ExprNode {
    if (typeof code === "number") {
        return code;
    }
    const tokens = lexn(code);
    const expr = parseStatements(0);
    if (expr === null) {
        throw new Error("Missing expression");
    }
    if (expr[1] < tokens.length) {
        console.log("bad tree:", "e0:", expr[0], "e1:", expr[1], "numTokens:", tokens.length, "token:", tokens[expr[1]])
        throw new Error(`Unexpected token: '${tokens[expr[1]]}' after parsing: ${JSON.stringify(expr[0])}`);
    }
    return expr[0];
    function parsePrimary(i: number): ParseState|null {
        if (i >= tokens.length) {
            return null;
        }
        const t = tokens[i];
        if (typeof t === "number") {
            return [t, i + 1];
        }
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
        if (!tokenIsIdent(t)) {
            return null;
        }
        let result: ParseState = [t, i + 1];
        if (i + 1 < tokens.length && tokens[i + 1] === "(") {
            const args: ExprNode[] = [];
            let j = i + 2;
            while (j < tokens.length) {
                const arg = parseExpr(j);
                if (arg === null) {
                    throw new Error("Missing argument");
                }
                args.push(arg[0]);
                j = arg[1];
                if (j >= tokens.length) {
                    throw new Error("Unexpected end of expression");
                }
                const t2 = tokens[j];
                if (typeof t2 !== "string") {
                    throw new Error("Expected , or )");
                }
                if (t2 === ")") {
                    break;
                }
                if (t2 !== ",") {
                    throw new Error("Expected ,");
                }
                j++;
            }
            const func = result[0];
            args.splice(0, 0, func);
            result = [["apply", args], j + 1];
        }
        return result;
    }
    function parseExpr(i: number): ParseState|null {
        return parseComparison(i);
    }
    function parseComparison(i: number): ParseState|null {
        let expr = parseAddOrSubtract(i);
        if (expr === null) {
            return null;
        }
        i = expr[1];
        while (i < tokens.length) {
            const t = tokens[i];
            if (typeof t !== "string") {
                break;
            }
            if (t === "==" || t === "!=" || t === "<" || t === ">" || t === "<=" || t === ">=") {
                const expr2 = parseAddOrSubtract(i + 1);
                if (expr2 === null) {
                    throw new Error("Missing expression after " + t);
                }
                expr = [[t, [expr[0], expr2[0]]], expr2[1]];
                i = expr2[1];
                continue;
            }
            break;
        }
        return expr;
    }
    function parseAddOrSubtract(i: number): ParseState|null {
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
    function parseMultiplyOrDivide(i: number): ParseState|null {
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
    function parseStatement(i: number): ParseState|null {
        if (i >= tokens.length) {
            return null;
        }
        
        // Empty statement?
        if (tokens[i] === ";") {
            // console.log("parse Empty Statement:", i, tokens.slice(i));
            return [["block", []], i + 1];
        }
        // Block statement?
        if (tokens[i] === "{") {
            // console.log("parse Block Statement:", i, tokens.slice(i));
            let j = i + 1;
            if (j >= tokens.length) {
                throw new Error("Missing } at end of input");
            }
            if (tokens[j] === "}") {
                // console.log("block statement empty");
                return [["block", []], j + 1];
            }
            const statements = parseStatements(j);
            if (statements === null) {
                throw new Error("Missing statements after {");
            }
            j = statements[1];
            // console.log("block statements end:", j, tokens[j]);
            if (tokens[j] !== "}") {
                throw new Error(`Missing } (${tokens})`);
            }
            // console.log("block statement end:", j + 1, tokens[j + 1]);
            const statementsNode: ExprNode = statements[0];
            if (statementsNode instanceof Array && statementsNode[0] === "statements") {
                return [["block", statementsNode[1]], j + 1];
            }
            else {
                return [["block", [statementsNode]], j + 1];
            }
        }
        // If statement?
        if (tokens[i] === "if") {
            // console.log("parse If Statement:", i, tokens.slice(i));
            let j = i + 1;
            if (j >= tokens.length) {
                throw new Error("Missing ( after if");
            }
            if (tokens[j] !== "(") {
                throw new Error("Missing ( after if");
            }
            const condExpr = parseExpr(j + 1);
            if (condExpr === null) {
                throw new Error("Missing condition after if");
            }
            j = condExpr[1];
            if (j >= tokens.length || tokens[j] !== ")") {
                throw new Error("Missing ) after if condition");
            }
            const thenExpr = parseStatement(j + 1);
            if (thenExpr === null) {
                throw new Error("Missing then after if condition");
            }
            j = thenExpr[1];
            if (j >= tokens.length || tokens[j] !== "else") {
                return [["if", [condExpr[0], thenExpr[0]]], j];
            }
            const elseExpr = parseStatement(j + 1);
            if (elseExpr === null) {
                throw new Error("Missing else after if condition");
            }
            j = elseExpr[1];
            return [["if", [condExpr[0], thenExpr[0], elseExpr[0]]], j];
        }
        // return statement?
        if (tokens[i] === "return") {
            // console.log("parse Return Statement:", i, tokens.slice(i));
            const returnExpr = parseExpr(i + 1);
            if (returnExpr === null) {
                throw new Error("Missing expression after return");
            }
            return [["return", [returnExpr[0]]], returnExpr[1]];
        }

        // Expression statement?
        const expr = parseExpr(i);
        if (expr === null) {
            // console.log("Failed to parse statement");
            return null;
        }
        // console.log("parse Expression Statement:", i, tokens.slice(i), expr);
        i = expr[1];
        if (i < tokens.length && tokens[i] === "=") {
            const expr2 = parseExpr(i + 1);
            if (expr2 === null) {
                throw new Error("Missing right hand side of assignment");
            }
            return [["assign", [expr[0], expr2[0]]], expr2[1]];
        }
        return expr;
    }
    function parseStatements(i: number): ParseState|null {
        if (i >= tokens.length) {
            return null;
        }
        // console.log("parseStatements:", i, tokens.slice(i));
        const expr = parseStatement(i);
        if (expr === null) {
            return null;
        }
        i = expr[1];
        if (i >= tokens.length) {
            return expr;
        }
        let children: ExprNode[] = [expr[0]];
        while (i < tokens.length) {
            const t = tokens[i];
            if (t !== ";") {
                break;
            }
            // console.log("got semi-colon at", i, tokens.slice(i));
            i += 1;
            const expr2 = parseStatement(i);
            if (expr2 === null) {
                // console.log("no more statements");
                break;
            }
            children.push(expr2[0]);
            i = expr2[1];
        }
        if (children.length === 1) {
            return [children[0], i];
        }
        // console.log("parsed statements", i, tokens.slice(i));
        return [["statements", children], i];
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
    if (ast[0] === "apply") {
        const f = ast[1][0];
        const fstr = exprNodeToString(f);
        const args = ast[1].slice(1);
        const argstrs = args.map(exprNodeToString);
        return `${fstr}(${argstrs.join(", ")})`;
    }
    if (ast[0] === "assign") {
        const lhs = ast[1][0];
        const rhs = ast[1][1];
        return `${exprNodeToString(lhs)} = ${exprNodeToString(rhs)}`;
    }
    if (ast[0] === "if") {
        const cond = ast[1][0];
        const then = ast[1][1];
        const else_ = ast[1][2];
        return `if (${exprNodeToString(cond)}) { ${exprNodeToString(then)} } else { ${exprNodeToString(else_)} }`;
    }

    throw new Error(`Unknown AST node type when printing: ${ast[0]}`);
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
    const expr = parseCode(code);
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
