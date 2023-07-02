/*======== TEST FRAMEWORK ========*/

const testreg = [];
function test(description, callback) { testreg.push({ description, callback }); }

class Expect {
    constructor(value, truth) { this.value = value; this.truth = truth; }
    get toBeText() { return this.truth(true) ? "to be" : "to not be"; }
    toBe(expected) { if (this.truth(!Object.is(this.value, expected))) { throw new Error(`Expected «${this.value}» ${this.toBeText} «${expected}»`); } }
    toBeCloseTo(expected, precision) { const expDiff = Math.pow(10, -(precision||2))/2; if (this.truth(Math.abs(this.value - expected) >= expDiff)) { throw new Error(`Expected «${this.value}» ${this.toBeText} close to «${expected}» (diff: < ${expDiff.toFixed((precision||2)+1)})`); } }
    toBeGreaterThan(expected) { if (this.truth(!(this.value > expected))) { throw new Error(`Expected «${this.value}» ${this.toBeText} greater than «${expected}»`); } }
    toBeGreaterThanOrEqual(expected) { if (this.truth(!(this.value >= expected))) { throw new Error(`Expected «${this.value}» ${this.toBeText} greater than or equal to «${expected}»`); } }
    toBeInstanceOf(expected) { if (this.truth(!(this.value instanceof expected))) { throw new Error(`Expected «${this.value}» ${this.toBeText} instance of «${expected}»`); } }
    toBeLessThan(expected) { if (this.truth(!(this.value < expected))) { throw new Error(`Expected «${this.value}» ${this.toBeText} less than «${expected}»`); } }
    toBeLessThanOrEqual(expected) { if (this.truth(!(this.value <= expected))) { throw new Error(`Expected «${this.value}» ${this.toBeText} less than or equal to «${expected}»`); } }
    toBeNaN() { if (this.truth(!Number.isNaN(this.value))) { throw new Error(`Expected «${this.value}» ${this.toBeText} NaN`); } }
    toBeNull() { if (this.truth(this.value !== null)) { throw new Error(`Expected «${this.value}» ${this.toBeText} null`); } }
    toEqual(expected) { if (this.truth(!eq(this.value, expected))) { throw new Error(`Expected «${JSON.stringify(this.value)}» to equal «${JSON.stringify(expected)}»`); } }
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
    if (typeof a === 'number' && typeof b === 'number') {
        if (Number.isNaN(a) && Number.isNaN(b)) return true;
        if (a === 0 && b === 0) return true;
        const epsilon = Math.max(Math.abs(a), Math.abs(b)) * 1e-6;
        return Math.abs(a - b) < epsilon;
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

function makeStatus($testDiv) {
    const $status = document.createElement('div');
    $status.innerHTML = `<span class="success">0</span> succeeded / <span class="failure">0</span> failed of ${testreg.length} tests: <span class="status-text">Starting...</span>`;
    $testDiv.appendChild($status);
    const $success = $status.querySelector('.success');
    const $failure = $status.querySelector('.failure');
    const $statusText = $status.querySelector('.status-text');
    let numSuccess = 0;
    let numFailure = 0;
    const updateStatus = () => {
        const numRun = Math.max(1, numSuccess + numFailure);
        $success.innerText = `${numSuccess} (${(numSuccess/numRun*100).toFixed(1)}%)`;
        $failure.innerText = `${numFailure} (${(numFailure/numRun*100).toFixed(1)}%)`;
    };
    const setStatusText = text => { $statusText.innerText = text; };
    const setSucceeded = () => { numSuccess++; updateStatus(); };
    const setFailed = () => { numFailure++; updateStatus(); };
    return [setSucceeded, setFailed, setStatusText];
}

async function runTestsAsync($testDiv) {
    const [setSucceeded, setFailed, setStatusText] = makeStatus($testDiv);
    const $failureBody = makeTable('Failures', $testDiv);
    const $successBody = makeTable('Success', $testDiv);
    // Run the tests
    for (let t of testreg) {
        let error = undefined;
        try {
            setStatusText(t.description);
            let result = t.callback();
            if (result instanceof Promise) {
                await result;
            }
            appendTestResult($successBody, t);
            setSucceeded();
        } catch (e) {
            error = e;
            console.error(e);
            appendTestResult($failureBody, t, error);
            setFailed();
        }
        await new Promise(resolve => setTimeout(resolve, 1));
    }
    setStatusText('Done');
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
