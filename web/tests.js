"use strict";
/*
These are tests that do not work in the normal node test environment
because they require WebGPU.
*/
const tests = [];
function test(description, callback) { tests.push({ description, callback }); }
const tensor = torch.tensor;
const Tensor = torch.Tensor;
const functions = torch.functions;


test("webgpu is supported", () => {
    expect(torch.hasWebGPU()).toBe(true);
});

test("tensor is webgpu", async () => {
    const x = torch.tensor([[1, 2, 3], [4, 5, 6]]);
    expect(x.requiresGrad).toBe(false);
    expect(x.device.type).toBe("webgpu");
    expect(await x.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
});

test("add_ vectors", async () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([100, 200, 300]);
    const aa = a.add_(b);
    expect(await a.toArrayAsync()).toEqual([101, 202, 303]);
});

test("abs value", async () => {
    const x = tensor([[-1, 2, -3], [4, -5, 6]]);
    const y = x.abs();
    expect(await y.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
});

test("abs value with grad", async () => {
    const x = tensor({data:[[-1, 2, -3], [4, -5, 6]], requiresGrad:true});
    const y = x.abs();
    expect(await y.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(y.requiresGrad).toBe(true);
});

test("matrix multiply", async () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    const b = tensor([[7, 8], [9, 10], [11, 12]]);
    const c = a.mm(b);
    expect(await c.toArrayAsync()).toEqual([[58, 64], [139, 154]]);
});

test("linear forward", async () => {
    const input = tensor([[3]]);
    const weight = tensor([[10], [11]]);
    const bias = tensor([[1000, 10000]]);
    const output = functions.LinearFunction.apply(input, weight, bias);
    const expected = [[3 * 10 + 1000, 3 * 11 + 10000]];
    expect(output.shape).toEqual([1, 2]);
    expect(await output.toArrayAsync()).toEqual(expected);
});

test("abs backwards", async () => {
    const input = new Tensor({data:[-3], requiresGrad:true});
    const output = input.abs();
    output.backward();
    expect(input.grad).not.toBeNull();
    expect(output.grad).not.toBeNull();
    expect(await input.grad.toArrayAsync()).toEqual([-1]);
});

test("linear backwards", () => {
    const input = new Tensor([[3]], "float32", null, true);
    const weight = new Tensor([[10], [11]], "float32", null, true);
    const bias = new Tensor([[1000, 10000]], "float32", null, true);
    const output = functions.LinearFunction.apply(input, weight, bias);
    expect(output).toBeInstanceOf(Tensor);
    expect(output.shape).toEqual([1, 2]);
    // const loss = output.sum();
    // expect(loss.gradFunc).not.toBeNull();
    // expect(loss.requiresGrad).toBe(true);
    // loss.backward();
    // expect(input.grad).not.toBeNull();
    // expect(weight.grad).not.toBeNull();
    // expect(bias.grad).not.toBeNull();
    // expect(output.grad).not.toBeNull();
});




/*======== TEST FRAMEWORK ========*/

class Expect {
    constructor(value, truth) { this.value = value; this.truth = truth; }
    toBe(expected) { if (this.truth(!Object.is(this.value, expected))) { throw new Error(`Expected «${this.value}» to be «${expected}»`); } }
    toBeGreaterThan(expected) { if (this.truth(!(this.value > expected))) { throw new Error(`Expected «${this.value}» to be greater than «${expected}»`); } }
    toBeInstanceOf(expected) { if (this.truth(!(this.value instanceof expected))) { throw new Error(`Expected «${this.value}» to be instance of «${expected}»`); } }
    toBeNull() { if (this.truth(this.value !== null)) { throw new Error(`Expected «${this.value}» to be null`); } }
    toEqual(expected) { if (this.truth(!eq(this.value, expected))) { throw new Error(`Expected «${this.value}» to equal «${expected}»`); } }
    toThrow(expected) {
        try { this.value(); } catch (e) {
            if (this.truth(true))
                return;
        } if (this.truth(true))
            throw new Error(`Expected ${this.value} to throw`);
    }
    get not() { return new Expect(this.value, t => !this.truth(t)); };
}

function expect(value) { return new Expect(value, t => t); }

function eq(a, b) {
    if (a === b) return true;
    if (a instanceof Array && b instanceof Array) {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) { if (!eq(a[i], b[i])) return false; }
        return true;
    }
    return false;
}

async function runTestsAsync($testDiv) {
    // Build the UI table
    const $testTable = document.createElement('table');
    $testTable.className = 'tests';
    $testDiv.appendChild($testTable);
    const $testTableHead = document.createElement('thead');
    $testTable.appendChild($testTableHead);
    const $testTableHeadRow = document.createElement('tr');
    $testTableHead.appendChild($testTableHeadRow);
    for (let h of ['Test', 'Error']) {
        const $testTableHeadCell = document.createElement('th');
        $testTableHeadCell.innerText = h;
        $testTableHeadRow.appendChild($testTableHeadCell);
    }
    const $testTableBody = document.createElement('tbody');
    $testTable.appendChild($testTableBody);

    // Run the tests
    for (let t of tests) {
        let error = undefined;
        try {
            let result = t.callback();
            if (result instanceof Promise) {
                await result;
            }
        } catch (e) {
            error = e;
            console.error(e);
        }
        const $testRow = document.createElement('tr');
        $testRow.className = error ? 'error' : 'success';
        $testTableBody.appendChild($testRow);
        const $testDescriptionCell = document.createElement('td');
        $testDescriptionCell.innerText = t.description;
        $testRow.appendChild($testDescriptionCell);
        const $testErrorCell = document.createElement('td');
        if (error) {
            $testErrorCell.innerText = error.message;
            const $stackDiv = document.createElement('div');
            $stackDiv.className = 'stack';
            $stackDiv.innerText = error.stack;
            $testErrorCell.appendChild($stackDiv);
        }
        $testRow.appendChild($testErrorCell);
    }
}
