let seed = 42;

export function getSeed(): number {
    const oldSeed = seed;
    seed = (Math.random() * 0x7FFFFFFF) >>> 0;
    return oldSeed;
}
