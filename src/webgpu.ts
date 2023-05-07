export function hasWebGPU() {
    const anavigator = navigator as any;
    if (!anavigator.gpu) { return false; }
    return true;
}
