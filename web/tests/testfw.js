/*======== TEST FRAMEWORK ========*/

const testreg = [];
function test(description, callback) { testreg.push({ description, callback }); }

class Expect {
    constructor(value, truth) { this.value = value; this.truth = truth; }
    toBe(expected) { if (this.truth(!Object.is(this.value, expected))) { throw new Error(`Expected «${this.value}» to be «${expected}»`); } }
    toBeCloseTo(expected, precision) { const expDiff = Math.pow(10, -precision)/2; if (this.truth(Math.abs(this.value - expected) >= expDiff)) { throw new Error(`Expected «${this.value}» to be close to «${expected}» (diff: < ${expDiff.toFixed(precision+1)})`); } }
    toBeGreaterThan(expected) { if (this.truth(!(this.value > expected))) { throw new Error(`Expected «${this.value}» to be greater than «${expected}»`); } }
    toBeInstanceOf(expected) { if (this.truth(!(this.value instanceof expected))) { throw new Error(`Expected «${this.value}» to be instance of «${expected}»`); } }
    toBeNaN() { if (this.truth(!Number.isNaN(this.value))) { throw new Error(`Expected «${this.value}» to be NaN`); } }
    toBeNull() { if (this.truth(this.value !== null)) { throw new Error(`Expected «${this.value}» to be null`); } }
    toEqual(expected) { if (this.truth(!eq(this.value, expected))) { throw new Error(`Expected «${this.value}» to equal «${expected}»`); } }
    toHaveLength(expected) { if (this.truth(this.value.length !== expected)) { throw new Error(`Expected «${this.value}» to have length «${expected}»`); } }
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
    if (a instanceof Object && b instanceof Object) {
        if (Object.keys(a).length !== Object.keys(b).length) return false;
        for (let k in a) { if (!eq(a[k], b[k])) return false; }
        return true;
    }
    return false;
}

function makeTable(name, $testDiv) {
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
    return $testTableBody;
}

async function runTestsAsync($testDiv) {
    const $failureBody = makeTable('Failures', $testDiv);
    const $successBody = makeTable('Success', $testDiv);
    // Run the tests
    for (let t of testreg) {
        let error = undefined;
        try {
            let result = t.callback();
            if (result instanceof Promise) {
                await result;
            }
            appendTestResult($successBody, t);
        } catch (e) {
            error = e;
            console.error(e);
            appendTestResult($failureBody, t, error);
        }
    }
}

function appendTestResult($testTableBody, t, error) {
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
