const path = require("path");
const TerserPlugin = require("terser-webpack-plugin");

module.exports = {
    entry: "./src/torch.ts", // Set the entry TypeScript file
    devtool: "source-map",
    optimization: {
        minimizer: [
            new TerserPlugin({
                terserOptions: {
                    keep_fnames: true, // Keep function names
                },
            }),
        ],
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: "ts-loader",
                exclude: /node_modules/,
            },
        ],
    },
    resolve: {
        extensions: [".tsx", ".ts", ".js"],
    },
    // rename index.d.ts to torch.d.ts

    output: {
        path: path.resolve(__dirname, "dist"),
        filename: "torch.js", // Set the output file name
        library: "torch", // Set the exported library name
        libraryTarget: "umd", // Set the target type of the exported library
        globalObject: "torch", // Set the global object name
    },
};
