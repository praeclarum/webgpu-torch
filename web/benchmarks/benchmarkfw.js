/*======== BENCHMARK FRAMEWORK ========*/


function getInputPermutations(benchmarkInputs) {
    function getRemainingPerms(inputs, perms) {
        const nextInputIndex = inputs.length;
        if (nextInputIndex === benchmarkInputs.length) {
            perms.push(inputs);
            return;
        }
        for (let i = 0; i < benchmarkInputs[nextInputIndex].values.length; i++) {
            const inputsCopy = inputs.slice();
            inputsCopy.push(benchmarkInputs[nextInputIndex].values[i]);
            getRemainingPerms(inputsCopy, perms);
        }
    }
    const perms = [];
    getRemainingPerms([], perms);
    return perms;
}

async function runUnaryBenchmarkAsync(benchmark, inputs) {
    const shape = inputs[0];
    const operation = torch[inputs[1]];
    async function runIteration() {
        const x = torch.ones(shape);
        const start = performance.now();
        const y = operation(x);
        const yar = await y.toArrayAsync();
        const end = performance.now();
        return end - start;
    }
    for (let i = 0; i < benchmark.warmupIterations; i++) {
        await runIteration();
    }
    const ms = [];
    for (let i = 0; i < benchmark.iterations; i++) {
        ms.push(await runIteration());
    }
    return ms;
}

async function runBenchmarkAsync(benchmark, inputs) {
    if (benchmark.type === 'unary') {
        return await runUnaryBenchmarkAsync(benchmark, inputs);
    }
    else {
        throw `Unknown benchmark type '${benchmark.type}'`;
    }
}

async function runBenchmarksAsync($benchmarksDiv) {
    // Load the test JSON
    const jsonUrl = "./benchmarks.json";
    const response = await fetch(jsonUrl);
    const benchmarks = await response.json();

    // Create the UI
    const $benchmarksTable = document.createElement('table');
    $benchmarksTable.className = 'benchmarks';
    $benchmarksDiv.appendChild($benchmarksTable);
    const $benchmarksTableHead = document.createElement('thead');
    $benchmarksTable.appendChild($benchmarksTableHead);
    const $benchmarksTableHeadRow = document.createElement('tr');
    $benchmarksTableHead.appendChild($benchmarksTableHeadRow);
    for (let h of ['Benchmark', 'Time (ms)', 'Error']) {
        const $benchmarksTableHeadCell = document.createElement('th');
        $benchmarksTableHeadCell.innerText = h;
        $benchmarksTableHeadRow.appendChild($benchmarksTableHeadCell);
    }
    const $benchmarksTableBody = document.createElement('tbody');
    $benchmarksTable.appendChild($benchmarksTableBody);

    // Run the benchmarks
    const benchmarkResults = [];
    for (let b of benchmarks.benchmarks) {
        
        const inputPerms = getInputPermutations(b.inputs);

        for (let ip of inputPerms) {
            const result = { benchmark: b, inputs: ip, meanTime: 0.0 };
            try {
                let times = await runBenchmarkAsync(b, ip);
                const meanTime = times.reduce((a, b) => a + b, 0) / times.length;
                result.meanTime = meanTime;
            }
            catch (e) {
                result.error = e;
                console.error(e);
            }
            benchmarkResults.push(result);
            appendBenchmarkResult(result);
        }
    }

    function appendBenchmarkResult(b) {
        const $benchmarksTableRow = document.createElement('tr');
        $benchmarksTableBody.appendChild($benchmarksTableRow);
        const $benchmarksTableRowName = document.createElement('td');
        $benchmarksTableRowName.innerText = `${b.benchmark.name}(${b.inputs.join(', ')}))`;
        $benchmarksTableRow.appendChild($benchmarksTableRowName);
        const $benchmarksTableRowTime = document.createElement('td');
        $benchmarksTableRowTime.innerText = `${b.meanTime.toFixed(2)}`;
        $benchmarksTableRow.appendChild($benchmarksTableRowTime);
        const $benchmarksTableRowError = document.createElement('td');
        $benchmarksTableRowError.innerText = b.error ? `${b.error}` : '';
        $benchmarksTableRow.appendChild($benchmarksTableRowError);
    }
}
