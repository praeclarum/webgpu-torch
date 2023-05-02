"use strict";
const tests = [];
function test(description, callback) { tests.push({ description, callback }); }
const torch = jstorch;


test("webgpu is supported", () => {
    expect(torch.hasWebGPU()).toBe(true);
});

test("tensor is webgpu", () => {
    const x = torch.tensor([[1, 2, 3], [4, 5, 6]], "float32");
    expect(x.device.type).toBe("webgpu");
});

test("linear forward", () => {
    const x = torch.tensor([[1, 2, 3], [4, 5, 6]], "float32");
    console.log(x);
    expect(x.shape).toBe([2, 3]);
    const w = torch.tensor([[100, 200, 300]], "float32", true);
    const b = torch.tensor([[-1000], [1000]], "float32", true);
    const y = torch.LinearFunction.apply(x, w, b);
    expect(y.shape).toBe([2, 1]);
});



/*======== TEST FRAMEWORK ========*/

function Expect(value) { this.value = value; }
Expect.prototype.toBe = function(expected) { if (!Object.is(this.value, expected)) { throw new Error(`Expected ${this.value} to be ${expected}`); } };
Expect.prototype.toBeInstanceOf = function(expected) { if (!(this.value instanceof expected)) { throw new Error(`Expected ${this.value} to be instance of ${expected}`); } };
Expect.prototype.toEqual = function(expected) { if (!eq(this.value, expected)) { throw new Error(`Expected ${this.value} to equal ${expected}`); } };
function expect(value) { return new Expect(value); }

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
        }
        $testRow.appendChild($testErrorCell);
    }
}
