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
    const x = torch.ones(shape);
    // const y = torch.zeros(shape);
    async function runIterationAsync() {
        // console.time('ones');
        // console.timeEnd('ones');
        const start = performance.now();
        let y = x;
        // const y = torch.zeros(shape);
        {
            for (let i = 0; i < benchmark.depth; i++) {
                // operation(x, y);
                y = operation(x);
            }
        }
        // console.log();
        const yar = await y.storage.toTypedArrayAsync(y.dtype);
        // if (yar.length < 1000) {
        //     console.log("y1", yar);
        // }
        // await y.device._device.queue.onSubmittedWorkDone();
        const end = performance.now();
        return (end - start) / benchmark.depth;
    }
    for (let i = 0; i < benchmark.warmupIterations; i++) {
        await runIterationAsync();
        await new Promise(resolve => setTimeout(resolve, 20));
    }
    const ms = [];
    for (let i = 0; i < benchmark.iterations; i++) {
        ms.push(await runIterationAsync());
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

async function loadJsonAsync(url) {
    const response = await fetch(url);
    const o = await response.json();
    return o;
}

async function runBenchmarksAsync($benchmarksDiv) {
    // Load the test JSON
    const benchmarks = await loadJsonAsync("./benchmarks.json");
    const otherResults = await loadJsonAsync("./results.json");

    const otherKeys = Object.keys(otherResults);

    // Create the UI
    const $benchmarksTable = document.createElement('table');
    $benchmarksTable.className = 'benchmarks';
    $benchmarksDiv.appendChild($benchmarksTable);
    const $benchmarksTableHead = document.createElement('thead');
    $benchmarksTable.appendChild($benchmarksTableHead);
    const $benchmarksTableHeadRow = document.createElement('tr');
    $benchmarksTableHead.appendChild($benchmarksTableHeadRow);
    const hs = ['Benchmark', 'Time (ms)'];
    for (let ok of otherKeys) {
        hs.push(otherResults[ok].device_name);
    }
    hs.push('Error');
    for (let h of hs) {
        const $benchmarksTableHeadCell = document.createElement('th');
        $benchmarksTableHeadCell.innerText = h;
        $benchmarksTableHeadRow.appendChild($benchmarksTableHeadCell);
    }
    const $benchmarksTableBody = document.createElement('tbody');
    $benchmarksTable.appendChild($benchmarksTableBody);

    const repr = (v) => {
        if (typeof v === 'string') {
            return `'${v}'`;
        }
        else if (typeof v === 'number') {
            return `${v}`;
        }
        throw `Unknown type '${typeof v}'`;
    };

    // Run the benchmarks
    const benchmarkResults = [];
    for (let b of benchmarks.benchmarks) {
        
        const inputPerms = getInputPermutations(b.inputs);

        for (let ip of inputPerms) {
            const benchmarkKey = `${b.name}(${ip.map(v => repr(v)).join(', ')})`;
            const result = { key: benchmarkKey, inputs: ip, meanTime: 0.0 };
            for (let ok of otherKeys) {
                if (otherResults[ok].results[benchmarkKey]) {
                    result[ok] = otherResults[ok].results[benchmarkKey].mean_ms;
                }
            }
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
            await new Promise((resolve, reject) => {
                setTimeout(() => {
                    resolve();
                }, 10);
            });
        }
    }

    function appendBenchmarkResult(b) {
        const $benchmarksTableRow = document.createElement('tr');
        $benchmarksTableBody.appendChild($benchmarksTableRow);
        const $benchmarksTableRowName = document.createElement('td');
        $benchmarksTableRowName.innerText = b.key;
        $benchmarksTableRow.appendChild($benchmarksTableRowName);
        const $benchmarksTableRowTime = document.createElement('td');
        $benchmarksTableRowTime.innerText = `${b.meanTime.toFixed(3)}`;
        $benchmarksTableRow.appendChild($benchmarksTableRowTime);
        for (let ok of otherKeys) {
            const $benchmarksTableRowOther = document.createElement('td');
            if (otherResults[ok].results[b.key]) {
                const ms = otherResults[ok].results[b.key].mean_ms;
                $benchmarksTableRowOther.innerText = `${ms.toFixed(3)}`;
            }
            $benchmarksTableRow.appendChild($benchmarksTableRowOther);
        }
        const $benchmarksTableRowError = document.createElement('td');
        $benchmarksTableRowError.innerText = b.error ? `${b.error}` : '';
        $benchmarksTableRow.appendChild($benchmarksTableRowError);
    }
}
