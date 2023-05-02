"use strict";
/*
These are tests that do not work in the normal node test environment
because they require WebGPU.
*/
const tests = [];
function test(description, callback) { tests.push({ description, callback }); }
const torch = jstorch;


test("webgpu is supported", () => {
    expect(torch.hasWebGPU()).toBe(true);
});

test("tensor is webgpu", () => {
    const x = torch.tensor([[1, 2, 3], [4, 5, 6]]);
    expect(x.requiresGrad).toBe(false);
    expect(x.device.type).toBe("webgpu");
});

test("linear forward", () => {
    const x = torch.tensor([[1, 2, 3], [4, 5, 6]]);
    expect(x.shape).toEqual([2, 3]);
    const w = torch.tensor({data: [[100, 200, 300]], requiresGrad: true});
    expect(w.requiresGrad).toBe(true);
    const b = torch.tensor({data: [[-1000], [1000]], requiresGrad: true});
    const y = torch.LinearFunction.apply(x, w, b);
    expect(y.shape).toEqual([2, 1]);
    expect(w.requiresGrad).toBe(true);
});



/*======== TEST FRAMEWORK ========*/

class Expect {
    constructor(value, truth) { this.value = value; this.truth = truth; }
    toBe(expected) { if (this.truth(!Object.is(this.value, expected))) { throw new Error(`Expected ${this.value} to be ${expected}`); } }
    toBeInstanceOf(expected) { if (this.truth(!(this.value instanceof expected))) { throw new Error(`Expected ${this.value} to be instance of ${expected}`); } }
    toBeNull() { if (this.truth(this.value !== null)) { throw new Error(`Expected ${this.value} to be null`); } }
    toEqual(expected) { if (this.truth(!eq(this.value, expected))) { throw new Error(`Expected ${this.value} to equal ${expected}`); } }
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
