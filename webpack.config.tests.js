const path = require("path");

module.exports = {
  mode: "development",
  entry: {
    tests: "./src/index_tests.ts"
  },
  output: {
    path: path.join(__dirname, "web", "tests"),
    filename: "tests.js",
    library: "tests",
    libraryTarget: "umd",
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/
      }
    ]
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js']
  },
  devtool: "source-map"
};
