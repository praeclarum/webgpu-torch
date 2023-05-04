const path = require("path");

module.exports = {
  entry: "./src/index.ts", // Set the entry TypeScript file
  devtool: "source-map",
  optimization: {
    minimize: false,
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
  output: {
    filename: "torch.js", // Set the output file name
    path: path.resolve(__dirname, "dist"),
    library: "torch",
    libraryTarget: "umd",
  },
};
