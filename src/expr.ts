// "Any sufficiently complicated C or Fortran program contains an ad hoc,
// informally-specified, bug-ridden, slow implementation of half of Common Lisp."
// - Greenspun's Tenth Rule

export type ExprCode = number | string;

export type ExprNodeType = "apply" | "block" | "assign" | "if" | "negate" | "return" | "statements" | "var" | "+" | "-" | "*" | "/" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||" | "&" | "|" | "^" | "!" | "~" | "^" | "%" | "?";

export type ExprAtom = string | ManifestNumber;
export type ExprCell = [ExprNodeType, ExprNode[]];
export type ExprNode = ExprAtom | ExprCell;

export type ManifestNumberType = "intAbstract" | "floatAbstract";
export class ManifestNumber {
    type: ManifestNumberType;
    value: number;
    constructor(type: ManifestNumberType, value: number) {
        this.type = type;
        this.value = value;
    }
}

export type ParsedExpr = ExprNode;

export type EvalEnv = { [name: string]: number | string };

function lexn(code: string): (string | ManifestNumber)[] {
    const tokens: (string | ManifestNumber)[] = [];
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
        if (c == "&") {
            if (i + 1 < n && code[i + 1] == "&") {
                tokens.push("&&");
                i += 2;
                continue;
            }
            tokens.push("&");
            i++;
            continue;
        }
        if (c == "|") {
            if (i + 1 < n && code[i + 1] == "|") {
                tokens.push("||");
                i += 2;
                continue;
            }
            tokens.push("|");
            i++;
            continue;
        }
        if (c >= "0" && c <= "9") {
            let s = "";
            let hasDecimal = false;
            while (i < n && (code[i] >= "0" && code[i] <= "9" || code[i] === ".")) {
                if (code[i] === ".") {
                    if (hasDecimal) {
                        throw new Error("Invalid number");
                    }
                    hasDecimal = true;
                }
                s += code[i];
                i++;
            }
            let numType: ManifestNumberType = hasDecimal ? "floatAbstract" : "intAbstract";
            if (i < n && code[i] === "f") {
                i++;
                numType = "floatAbstract";
            }
            tokens.push(new ManifestNumber(numType, parseFloat(s)));
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

function tokenIsIdent(token: string | ManifestNumber): boolean {
    if (token instanceof ManifestNumber) {
        return false;
    }
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
type ParserResult = ParseState|null;
type Parser = (i: number) => ParserResult;

export function parseCode(code: ExprCode): ExprNode {
    if (typeof code === "number") {
        // Detect if code is an integer
        if (code % 1 === 0) {
            return new ManifestNumber("intAbstract", code);
        }
        return new ManifestNumber("floatAbstract", code);
    }
    const tokens = lexn(code);
    const parsePrimary: Parser = (i: number): ParserResult => {
        if (i >= tokens.length) {
            return null;
        }
        const t = tokens[i];
        if (t instanceof ManifestNumber) {
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
    function parseExpr(i: number): ParserResult {
        if (i >= tokens.length) {
            return null;
        }
        return parseConditional(i);
    }
    function parseConditional(i: number): ParseState|null {
        if (i >= tokens.length) {
            return null;
        }
        let expr = parseLogicalOr(i);
        if (expr === null) {
            return null;
        }
        i = expr[1];
        if (i >= tokens.length) {
            return expr;
        }
        const t = tokens[i];
        if (t !== "?") {
            return expr;
        }
        const expr2 = parseExpr(i + 1);
        if (expr2 === null) {
            throw new Error("Missing expression after ?");
        }
        i = expr2[1];
        if (i >= tokens.length) {
            throw new Error("Unexpected end of conditional expression");
        }
        const t2 = tokens[i];
        if (typeof t2 !== "string" || t2 !== ":") {
            throw new Error(`Expected ':', got ${t2}`);
        }
        const expr3 = parseConditional(i + 1);
        if (expr3 === null) {
            throw new Error(`Missing expression after : \`${code}\``);
        }
        return [[t, [expr[0], expr2[0], expr3[0]]], expr3[1]];
    }
    function genericParseSeparatedList(parseChild: Parser, seps: string[]): (i: number)=> ParseState|null {
        const sepIndex: {[name:string]: boolean} = {};
        for (const sep of seps) {
            sepIndex[sep] = true;
        }
        return (i: number) => {
            let expr = parseChild(i);
            if (expr === null) {
                return null;
            }
            i = expr[1];
            while (i < tokens.length) {
                const t = tokens[i];
                if (typeof t !== "string") {
                    break;
                }
                if (t in sepIndex) {
                    const expr2 = parseChild(i + 1);
                    if (expr2 === null) {
                        throw new Error("Missing expression after " + t);
                    }
                    expr = [[t as ExprNodeType, [expr[0], expr2[0]]], expr2[1]];
                    i = expr2[1];
                    continue;
                }
                break;
            }
            return expr;
        };
    }
    const parseMultiplyOrDivide = genericParseSeparatedList(parseUnary, ["*", "/"]);
    const parseAddOrSubtract =genericParseSeparatedList(parseMultiplyOrDivide, ["+", "-"]);
    const parseRelational = genericParseSeparatedList(parseAddOrSubtract, ["<", ">", "<=", ">="]);
    const parseEquality = genericParseSeparatedList(parseRelational, ["==", "!="]);
    const parseAnd = genericParseSeparatedList(parseEquality, ["&"]);
    const parseExclusiveOr = genericParseSeparatedList(parseAnd, ["|"]);
    const parseInclusiveOr = genericParseSeparatedList(parseExclusiveOr, ["|"]);
    const parseLogicalAnd = genericParseSeparatedList(parseInclusiveOr, ["&&"]);
    const parseLogicalOr = genericParseSeparatedList(parseLogicalAnd, ["||"]);
    function parseUnary(i: number): ParseState|null {
        if (i >= tokens.length) {
            return null;
        }
        const t = tokens[i];
        if (typeof t === "string") {
            if (t === "+" || t === "-") {
                const expr = parseUnary(i + 1);
                if (expr === null) {
                    throw new Error("Missing expression");
                }
                if (t === "+") {
                    return expr;
                }
                return [["negate", [expr[0]]], expr[1]];
            }
            if (t === "!") {
                const expr = parseUnary(i + 1);
                if (expr === null) {
                    throw new Error("Missing expression");
                }
                return [[t, [expr[0]]], expr[1]];
            }
        }
        return parsePrimary(i);
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
        // var statement?
        if (tokens[i] === "var") {
            // console.log("parse Var Statement:", i, tokens.slice(i));
            const name = tokens[i + 1];
            if (typeof name !== "string") {
                throw new Error("Missing variable name after var");
            }
            const varEquals = tokens[i + 2];
            if (varEquals !== "=") {
                throw new Error("Missing = after var");
            }
            const varInit = parseExpr(i + 3);
            if (varInit === null) {
                throw new Error("Missing initial value after var");
            }
            return [["var", [name, varInit[0]]], varInit[1]];
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
    const expr = parseStatements(0);
    if (expr === null) {
        throw new Error("Missing expression");
    }
    if (expr[1] < tokens.length) {
        throw new Error(`Unexpected token '${tokens[expr[1]]}' after parsing ${JSON.stringify(expr[0])} from tokens [${tokens.slice(0, expr[1]+1)}]`);
    }
    return expr[0];
}

export function substitute(ast: ExprNode, match: (node: ExprNode)=>boolean, replace: (node: ExprNode)=>ExprNode): ExprNode {
    if (ast instanceof ManifestNumber) {
        return ast;
    }
    if (typeof ast === "string") {
        return ast;
    }
    const newChildren: ExprNode[] = [];
    for (const child of ast[1]) {
        newChildren.push(substitute(child, match, replace));
    }
    ast = [ast[0], newChildren];
    if (match(ast)) {
        ast = replace(ast);
    }
    return ast;
}

export function substituteIdentifiers(ast: ExprNode, subs: { [name: string]: ExprNode }): ExprNode {
    if (ast instanceof ManifestNumber) {
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
    if (ast instanceof ManifestNumber) {
        let s = "";
        if (ast.type === "floatAbstract") {
            s = ast.value.toString();
            if (s.indexOf(".") === -1) {
                s += "f";
            }
        } else {
            s = ast.value.toString();
        }
        return s;
    }
    if (typeof ast === "string") {
        return ast;
    }
    const nodeType: ExprNodeType = ast[0];
    switch (nodeType) {
    case "+":
        return `(${exprNodeToString(ast[1][0])} + ${exprNodeToString(ast[1][1])})`;
    case "-":
        return `(${exprNodeToString(ast[1][0])} - ${exprNodeToString(ast[1][1])})`;
    case "*":
        return `(${exprNodeToString(ast[1][0])} * ${exprNodeToString(ast[1][1])})`;
    case "/":
        return `(${exprNodeToString(ast[1][0])} / ${exprNodeToString(ast[1][1])})`;
    case "%":
        return `(${exprNodeToString(ast[1][0])} % ${exprNodeToString(ast[1][1])})`;
    case "==":
        return `(${exprNodeToString(ast[1][0])} == ${exprNodeToString(ast[1][1])})`;
    case "!=":
        return `(${exprNodeToString(ast[1][0])} != ${exprNodeToString(ast[1][1])})`;
    case "<":
        return `(${exprNodeToString(ast[1][0])} < ${exprNodeToString(ast[1][1])})`;
    case ">":
        return `(${exprNodeToString(ast[1][0])} > ${exprNodeToString(ast[1][1])})`;
    case "<=":
        return `(${exprNodeToString(ast[1][0])} <= ${exprNodeToString(ast[1][1])})`;
    case ">=":
        return `(${exprNodeToString(ast[1][0])} >= ${exprNodeToString(ast[1][1])})`;
    case "&&":
        return `(${exprNodeToString(ast[1][0])} && ${exprNodeToString(ast[1][1])})`;
    case "||":
        return `(${exprNodeToString(ast[1][0])} || ${exprNodeToString(ast[1][1])})`;
    case "&":
        return `(${exprNodeToString(ast[1][0])} & ${exprNodeToString(ast[1][1])})`;
    case "|":
        return `(${exprNodeToString(ast[1][0])} | ${exprNodeToString(ast[1][1])})`;
    case "^":
        return `(${exprNodeToString(ast[1][0])} ^ ${exprNodeToString(ast[1][1])})`;
    case "&&":
        return `(${exprNodeToString(ast[1][0])} && ${exprNodeToString(ast[1][1])})`;
    case "||":
        return `(${exprNodeToString(ast[1][0])} || ${exprNodeToString(ast[1][1])})`;
    case "?":
        return `(${exprNodeToString(ast[1][0])} ? ${exprNodeToString(ast[1][1])} : ${exprNodeToString(ast[1][2])})`;
    case "apply":
        const f = ast[1][0];
        const fstr = exprNodeToString(f);
        const args = ast[1].slice(1);
        const argstrs = args.map(exprNodeToString);
        return `${fstr}(${argstrs.join(", ")})`;
    case "block":
        return `{ ${ast[1].map(exprNodeToString).join("; ")} }`;
    case "assign":
        const lhs = ast[1][0];
        const rhs = ast[1][1];
        return `${exprNodeToString(lhs)} = ${exprNodeToString(rhs)}`;
    case "if":
        const cond = ast[1][0];
        const then = ast[1][1];
        const else_ = ast[1][2];
        return `if (${exprNodeToString(cond)}) { ${exprNodeToString(then)} } else { ${exprNodeToString(else_)} }`;
    case "negate":
        return `(-${exprNodeToString(ast[1][0])})`;
    case "statements":
        return ast[1].map(exprNodeToString).join("; ");
    case "return":
        return `return ${exprNodeToString(ast[1][0])}`;
    case "var":
        return `var ${ast[1][0]} = ${exprNodeToString(ast[1][1])}`;
    default:
        throw new Error(`Unknown AST node type when printing: ${ast[0]}`);
    }
}

export function exprNodeToWebGLShader(ast: ExprNode): string {
    // Convert conditional into select
    ast = substitute(ast, (node) => {
        if (!(node instanceof Array)) {
            return false;
        }
        if (node[0] === "?") {
            return true;
        }
        return false;
    }, (node) => {
        node = node as ExprCell;
        const cond = node[1][0];
        const then = node[1][1];
        const else_ = node[1][2];
        return ["apply", ["select", else_, then, cond]];
    });
    return exprNodeToString(ast);
}

export function exprCodeToWebGLShader(code: ExprCode, identifierSubs?: {[ident: string]: ExprNode}): string {
    let ast = parseCode(code);
    if (identifierSubs) {
        ast = substituteIdentifiers(ast, identifierSubs);
    }
    return exprNodeToWebGLShader(ast);
}



/*== COMPILER ==*/

const OP_CONSTANT = 0;
const OP_POP = 1;
const OP_READ = 2;
const OP_WRITE = 3;
const OP_ADD = 4;
const OP_APPLY = 5;
const OP_SUBTRACT = 6;
const OP_MULTIPLY = 7;
const OP_DIVIDE = 8;

export type CompiledExpr = (env: { [name: string]: any }) => number;

export function compileCode(code: ExprCode): CompiledExpr {
    if (typeof code === "number") {
        return (env: EvalEnv) => code;
    }
    const expr = parseCode(code);
    // console.log("parsed", expr);
    const instructions: [number, (number|string|null)][] = [];
    function emit(node: ExprNode) {
        if (node instanceof ManifestNumber) {
            instructions.push([OP_CONSTANT, node.value]);
            return;
        }
        if (typeof node === "string") {
            instructions.push([OP_READ, node]);
        }
        else if (node[0] === "+") {
            emit(node[1][0]);
            emit(node[1][1]);
            instructions.push([OP_ADD, null]);
        }
        else if (node[0] === "-") {
            emit(node[1][0]);
            emit(node[1][1]);
            instructions.push([OP_SUBTRACT, null]);
        }
        else if (node[0] === "*") {
            emit(node[1][0]);
            emit(node[1][1]);
            instructions.push([OP_MULTIPLY, null]);
        }
        else if (node[0] === "/") {
            emit(node[1][0]);
            emit(node[1][1]);
            instructions.push([OP_DIVIDE, null]);
        }
        else if (node[0] === "apply") {
            const f = node[1][0];
            const args = node[1].slice(1);
            for (const arg of args) {
                emit(arg);
            }
            instructions.push([OP_APPLY, exprNodeToString(f)]);
        }
        else if (node[0] === "statements") {
            const children = node[1];
            let i = 0;
            for (const child of children) {
                emit(child);
                if (i < children.length - 1) {
                    instructions.push([OP_POP, null]);
                }
                i++;
            }
        }
        else if (node[0] === "var") {
            const name = node[1][0];
            const value = node[1][1];
            emit(value);
            instructions.push([OP_WRITE, exprNodeToString(name)]);
        }
        else {
            throw new Error(`Unknown compile node type ${node}`);
        }
    }
    emit(expr);
    // console.log("ops", ops);
    return (env: { [name: string]: any }) => {
        // Evaluate the ops
        const stack: number[] = [];
        for (const ins of instructions) {
            const op = ins[0];
            if (op === OP_CONSTANT) {
                stack.push(ins[1] as number);
            }
            else if (op === OP_POP) {
                stack.pop();
            }
            else if (op == OP_READ) {
                stack.push(env[ins[1] as string]);
            }
            else if (op == OP_WRITE) {
                const v = stack.pop() as number;
                env[ins[1] as string] = v;
                stack.push(v);
            }
            else if (op == OP_APPLY) {
                const f = env[ins[1] as string];
                if (typeof f !== "function") {
                    throw new Error(`Expected function for "${ins[1]}", got ${f}`);
                }
                const args = [];
                for (let i = 0; i < f.length; i++) {
                    args.push(stack.pop());
                }
                args.reverse();
                stack.push(f(...args));
            }
            else {
                const b = stack.pop() as number;
                const a = stack.pop() as number;
                switch (op) {
                    case OP_ADD:
                        stack.push(a + b);
                        break;
                    case OP_SUBTRACT:
                        stack.push(a - b);
                        break;
                    case OP_MULTIPLY:
                        stack.push(a * b);
                        break;
                    case OP_DIVIDE:
                        stack.push(a / b);
                        break;
                    default:
                        throw new Error(`Unknown op ${op}`);
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
