const path = require("path");
const TerserPlugin = require('terser-webpack-plugin');

module.exports = {
  mode: "production",
  optimization: {
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          keep_fnames: true, // Keep function names
        },
      }),
    ],
  },
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
