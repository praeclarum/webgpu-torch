import { runOpgenTest as t } from "./ops_opgen_test_support";
test("abs", async () => {
    await t("abs", [[-2]], [[2]], [[-1]], false);
    await t("abs", [[-1]], [[1]], [[-1]], false);
    await t("abs", [[-0.5]], [[0.5]], [[-1]], false);
    await t("abs", [[0]], [[0]], [[0]], false);
    await t("abs", [[0.5]], [[0.5]], [[1]], false);
    await t("abs", [[1]], [[1]], [[1]], false);
    await t("abs", [[2]], [[2]], [[1]], false);
});
test("abs_", async () => {
    await t("abs_", [[-2]], [[2]], [], false);
    await t("abs_", [[-1]], [[1]], [], false);
    await t("abs_", [[-0.5]], [[0.5]], [], false);
    await t("abs_", [[0]], [[0]], [], false);
    await t("abs_", [[0.5]], [[0.5]], [], false);
    await t("abs_", [[1]], [[1]], [], false);
    await t("abs_", [[2]], [[2]], [], false);
});
test("acos", async () => {
    await t("acos", [[-2]], [["NaN"]], [["NaN"]], false);
    await t("acos", [[-1]], [[3.1415927410125732]], [["-Inf"]], false);
    await t("acos", [[-0.5]], [[2.094395160675049]], [[-1.154700517654419]], false);
    await t("acos", [[0]], [[1.5707963705062866]], [[-1]], false);
    await t("acos", [[0.5]], [[1.0471975803375244]], [[-1.154700517654419]], false);
    await t("acos", [[1]], [[0]], [["-Inf"]], false);
    await t("acos", [[2]], [["NaN"]], [["NaN"]], false);
});
test("acos_", async () => {
    await t("acos_", [[-2]], [["NaN"]], [], false);
    await t("acos_", [[-1]], [[3.1415927410125732]], [], false);
    await t("acos_", [[-0.5]], [[2.094395160675049]], [], false);
    await t("acos_", [[0]], [[1.5707963705062866]], [], false);
    await t("acos_", [[0.5]], [[1.0471975803375244]], [], false);
    await t("acos_", [[1]], [[0]], [], false);
    await t("acos_", [[2]], [["NaN"]], [], false);
});
test("acosh", async () => {
    await t("acosh", [[-2]], [["NaN"]], [[0.5773502588272095]], false);
    await t("acosh", [[-1]], [["NaN"]], [["+Inf"]], false);
    await t("acosh", [[-0.5]], [["NaN"]], [["NaN"]], false);
    await t("acosh", [[0]], [["NaN"]], [["NaN"]], false);
    await t("acosh", [[0.5]], [["NaN"]], [["NaN"]], false);
    await t("acosh", [[1]], [[0]], [["+Inf"]], false);
    await t("acosh", [[2]], [[1.316957950592041]], [[0.5773502588272095]], false);
});
test("acosh_", async () => {
    await t("acosh_", [[-2]], [["NaN"]], [], false);
    await t("acosh_", [[-1]], [["NaN"]], [], false);
    await t("acosh_", [[-0.5]], [["NaN"]], [], false);
    await t("acosh_", [[0]], [["NaN"]], [], false);
    await t("acosh_", [[0.5]], [["NaN"]], [], false);
    await t("acosh_", [[1]], [[0]], [], false);
    await t("acosh_", [[2]], [[1.316957950592041]], [], false);
});
test("asin", async () => {
    await t("asin", [[-2]], [["NaN"]], [["NaN"]], false);
    await t("asin", [[-1]], [[-1.5707963705062866]], [["+Inf"]], false);
    await t("asin", [[-0.5]], [[-0.5235987901687622]], [[1.154700517654419]], false);
    await t("asin", [[0]], [[0]], [[1]], false);
    await t("asin", [[0.5]], [[0.5235987901687622]], [[1.154700517654419]], false);
    await t("asin", [[1]], [[1.5707963705062866]], [["+Inf"]], false);
    await t("asin", [[2]], [["NaN"]], [["NaN"]], false);
});
test("asin_", async () => {
    await t("asin_", [[-2]], [["NaN"]], [], false);
    await t("asin_", [[-1]], [[-1.5707963705062866]], [], false);
    await t("asin_", [[-0.5]], [[-0.5235987901687622]], [], false);
    await t("asin_", [[0]], [[0]], [], false);
    await t("asin_", [[0.5]], [[0.5235987901687622]], [], false);
    await t("asin_", [[1]], [[1.5707963705062866]], [], false);
    await t("asin_", [[2]], [["NaN"]], [], false);
});
test("asinh", async () => {
    await t("asinh", [[-2]], [[-1.4436354637145996]], [[0.4472135901451111]], false);
    await t("asinh", [[-1]], [[-0.8813735842704773]], [[0.7071067690849304]], false);
    await t("asinh", [[-0.5]], [[-0.4812118113040924]], [[0.8944271802902222]], false);
    await t("asinh", [[0]], [[0]], [[1]], false);
    await t("asinh", [[0.5]], [[0.4812118113040924]], [[0.8944271802902222]], false);
    await t("asinh", [[1]], [[0.8813735842704773]], [[0.7071067690849304]], false);
    await t("asinh", [[2]], [[1.4436354637145996]], [[0.4472135901451111]], false);
});
test("asinh_", async () => {
    await t("asinh_", [[-2]], [[-1.4436354637145996]], [], false);
    await t("asinh_", [[-1]], [[-0.8813735842704773]], [], false);
    await t("asinh_", [[-0.5]], [[-0.4812118113040924]], [], false);
    await t("asinh_", [[0]], [[0]], [], false);
    await t("asinh_", [[0.5]], [[0.4812118113040924]], [], false);
    await t("asinh_", [[1]], [[0.8813735842704773]], [], false);
    await t("asinh_", [[2]], [[1.4436354637145996]], [], false);
});
test("atan", async () => {
    await t("atan", [[-2]], [[-1.1071487665176392]], [[0.20000000298023224]], false);
    await t("atan", [[-1]], [[-0.7853981256484985]], [[0.5]], false);
    await t("atan", [[-0.5]], [[-0.46364760398864746]], [[0.800000011920929]], false);
    await t("atan", [[0]], [[0]], [[1]], false);
    await t("atan", [[0.5]], [[0.46364760398864746]], [[0.800000011920929]], false);
    await t("atan", [[1]], [[0.7853981256484985]], [[0.5]], false);
    await t("atan", [[2]], [[1.1071487665176392]], [[0.20000000298023224]], false);
});
test("atan_", async () => {
    await t("atan_", [[-2]], [[-1.1071487665176392]], [], false);
    await t("atan_", [[-1]], [[-0.7853981256484985]], [], false);
    await t("atan_", [[-0.5]], [[-0.46364760398864746]], [], false);
    await t("atan_", [[0]], [[0]], [], false);
    await t("atan_", [[0.5]], [[0.46364760398864746]], [], false);
    await t("atan_", [[1]], [[0.7853981256484985]], [], false);
    await t("atan_", [[2]], [[1.1071487665176392]], [], false);
});
test("ceil", async () => {
    await t("ceil", [[-2]], [[-2]], [[0]], false);
    await t("ceil", [[-1]], [[-1]], [[0]], false);
    await t("ceil", [[-0.5]], [[0]], [[0]], false);
    await t("ceil", [[0]], [[0]], [[0]], false);
    await t("ceil", [[0.5]], [[1]], [[0]], false);
    await t("ceil", [[1]], [[1]], [[0]], false);
    await t("ceil", [[2]], [[2]], [[0]], false);
});
test("ceil_", async () => {
    await t("ceil_", [[-2]], [[-2]], [], false);
    await t("ceil_", [[-1]], [[-1]], [], false);
    await t("ceil_", [[-0.5]], [[0]], [], false);
    await t("ceil_", [[0]], [[0]], [], false);
    await t("ceil_", [[0.5]], [[1]], [], false);
    await t("ceil_", [[1]], [[1]], [], false);
    await t("ceil_", [[2]], [[2]], [], false);
});
test("cos", async () => {
    await t("cos", [[-2]], [[-0.416146844625473]], [[0.9092974066734314]], false);
    await t("cos", [[-1]], [[0.5403023362159729]], [[0.8414709568023682]], false);
    await t("cos", [[-0.5]], [[0.8775825500488281]], [[0.4794255495071411]], false);
    await t("cos", [[0]], [[1]], [[0]], false);
    await t("cos", [[0.5]], [[0.8775825500488281]], [[-0.4794255495071411]], false);
    await t("cos", [[1]], [[0.5403023362159729]], [[-0.8414709568023682]], false);
    await t("cos", [[2]], [[-0.416146844625473]], [[-0.9092974066734314]], false);
});
test("cos_", async () => {
    await t("cos_", [[-2]], [[-0.416146844625473]], [], false);
    await t("cos_", [[-1]], [[0.5403023362159729]], [], false);
    await t("cos_", [[-0.5]], [[0.8775825500488281]], [], false);
    await t("cos_", [[0]], [[1]], [], false);
    await t("cos_", [[0.5]], [[0.8775825500488281]], [], false);
    await t("cos_", [[1]], [[0.5403023362159729]], [], false);
    await t("cos_", [[2]], [[-0.416146844625473]], [], false);
});
test("cosh", async () => {
    await t("cosh", [[-2]], [[3.762195587158203]], [[-3.6268603801727295]], false);
    await t("cosh", [[-1]], [[1.5430806875228882]], [[-1.175201177597046]], false);
    await t("cosh", [[-0.5]], [[1.1276259422302246]], [[-0.5210952758789062]], false);
    await t("cosh", [[0]], [[1]], [[0]], false);
    await t("cosh", [[0.5]], [[1.1276259422302246]], [[0.5210952758789062]], false);
    await t("cosh", [[1]], [[1.5430806875228882]], [[1.175201177597046]], false);
    await t("cosh", [[2]], [[3.762195587158203]], [[3.6268603801727295]], false);
});
test("cosh_", async () => {
    await t("cosh_", [[-2]], [[3.762195587158203]], [], false);
    await t("cosh_", [[-1]], [[1.5430806875228882]], [], false);
    await t("cosh_", [[-0.5]], [[1.1276259422302246]], [], false);
    await t("cosh_", [[0]], [[1]], [], false);
    await t("cosh_", [[0.5]], [[1.1276259422302246]], [], false);
    await t("cosh_", [[1]], [[1.5430806875228882]], [], false);
    await t("cosh_", [[2]], [[3.762195587158203]], [], false);
});
test("deg2rad", async () => {
    await t("deg2rad", [[-2]], [[-0.03490658476948738]], [[0.01745329238474369]], false);
    await t("deg2rad", [[-1]], [[-0.01745329238474369]], [[0.01745329238474369]], false);
    await t("deg2rad", [[-0.5]], [[-0.008726646192371845]], [[0.01745329238474369]], false);
    await t("deg2rad", [[0]], [[0]], [[0.01745329238474369]], false);
    await t("deg2rad", [[0.5]], [[0.008726646192371845]], [[0.01745329238474369]], false);
    await t("deg2rad", [[1]], [[0.01745329238474369]], [[0.01745329238474369]], false);
    await t("deg2rad", [[2]], [[0.03490658476948738]], [[0.01745329238474369]], false);
});
test("deg2rad_", async () => {
    await t("deg2rad_", [[-2]], [[-0.03490658476948738]], [], false);
    await t("deg2rad_", [[-1]], [[-0.01745329238474369]], [], false);
    await t("deg2rad_", [[-0.5]], [[-0.008726646192371845]], [], false);
    await t("deg2rad_", [[0]], [[0]], [], false);
    await t("deg2rad_", [[0.5]], [[0.008726646192371845]], [], false);
    await t("deg2rad_", [[1]], [[0.01745329238474369]], [], false);
    await t("deg2rad_", [[2]], [[0.03490658476948738]], [], false);
});
test("exp", async () => {
    await t("exp", [[-2]], [[0.1353352814912796]], [[0.1353352814912796]], false);
    await t("exp", [[-1]], [[0.3678794503211975]], [[0.3678794503211975]], false);
    await t("exp", [[-0.5]], [[0.6065306663513184]], [[0.6065306663513184]], false);
    await t("exp", [[0]], [[1]], [[1]], false);
    await t("exp", [[0.5]], [[1.6487212181091309]], [[1.6487212181091309]], false);
    await t("exp", [[1]], [[2.7182817459106445]], [[2.7182817459106445]], false);
    await t("exp", [[2]], [[7.389056205749512]], [[7.389056205749512]], false);
});
test("exp_", async () => {
    await t("exp_", [[-2]], [[0.1353352814912796]], [], false);
    await t("exp_", [[-1]], [[0.3678794503211975]], [], false);
    await t("exp_", [[-0.5]], [[0.6065306663513184]], [], false);
    await t("exp_", [[0]], [[1]], [], false);
    await t("exp_", [[0.5]], [[1.6487212181091309]], [], false);
    await t("exp_", [[1]], [[2.7182817459106445]], [], false);
    await t("exp_", [[2]], [[7.389056205749512]], [], false);
});
test("exp2", async () => {
    await t("exp2", [[-2]], [[0.25]], [[0.1732867956161499]], false);
    await t("exp2", [[-1]], [[0.5]], [[0.3465735912322998]], false);
    await t("exp2", [[-0.5]], [[0.7071067690849304]], [[0.4901290535926819]], false);
    await t("exp2", [[0]], [[1]], [[0.6931471824645996]], false);
    await t("exp2", [[0.5]], [[1.4142135381698608]], [[0.9802581071853638]], false);
    await t("exp2", [[1]], [[2]], [[1.3862943649291992]], false);
    await t("exp2", [[2]], [[4]], [[2.7725887298583984]], false);
});
test("exp2_", async () => {
    await t("exp2_", [[-2]], [[0.25]], [], false);
    await t("exp2_", [[-1]], [[0.5]], [], false);
    await t("exp2_", [[-0.5]], [[0.7071067690849304]], [], false);
    await t("exp2_", [[0]], [[1]], [], false);
    await t("exp2_", [[0.5]], [[1.4142135381698608]], [], false);
    await t("exp2_", [[1]], [[2]], [], false);
    await t("exp2_", [[2]], [[4]], [], false);
});
test("expm1", async () => {
    await t("expm1", [[-2]], [[-0.8646647334098816]], [[0.1353352665901184]], false);
    await t("expm1", [[-1]], [[-0.6321205496788025]], [[0.3678794503211975]], false);
    await t("expm1", [[-0.5]], [[-0.39346933364868164]], [[0.6065306663513184]], false);
    await t("expm1", [[0]], [[0]], [[1]], false);
    await t("expm1", [[0.5]], [[0.6487212777137756]], [[1.6487212181091309]], false);
    await t("expm1", [[1]], [[1.718281865119934]], [[2.7182817459106445]], false);
    await t("expm1", [[2]], [[6.389056205749512]], [[7.389056205749512]], false);
});
test("expm1_", async () => {
    await t("expm1_", [[-2]], [[-0.8646647334098816]], [], false);
    await t("expm1_", [[-1]], [[-0.6321205496788025]], [], false);
    await t("expm1_", [[-0.5]], [[-0.39346933364868164]], [], false);
    await t("expm1_", [[0]], [[0]], [], false);
    await t("expm1_", [[0.5]], [[0.6487212777137756]], [], false);
    await t("expm1_", [[1]], [[1.718281865119934]], [], false);
    await t("expm1_", [[2]], [[6.389056205749512]], [], false);
});
test("floor", async () => {
    await t("floor", [[-2]], [[-2]], [[0]], false);
    await t("floor", [[-1]], [[-1]], [[0]], false);
    await t("floor", [[-0.5]], [[-1]], [[0]], false);
    await t("floor", [[0]], [[0]], [[0]], false);
    await t("floor", [[0.5]], [[0]], [[0]], false);
    await t("floor", [[1]], [[1]], [[0]], false);
    await t("floor", [[2]], [[2]], [[0]], false);
});
test("floor_", async () => {
    await t("floor_", [[-2]], [[-2]], [], false);
    await t("floor_", [[-1]], [[-1]], [], false);
    await t("floor_", [[-0.5]], [[-1]], [], false);
    await t("floor_", [[0]], [[0]], [], false);
    await t("floor_", [[0.5]], [[0]], [], false);
    await t("floor_", [[1]], [[1]], [], false);
    await t("floor_", [[2]], [[2]], [], false);
});
test("frac", async () => {
    await t("frac", [[-2]], [[0]], [[1]], false);
    await t("frac", [[-1]], [[0]], [[1]], false);
    await t("frac", [[-0.5]], [[-0.5]], [[1]], false);
    await t("frac", [[0]], [[0]], [[1]], false);
    await t("frac", [[0.5]], [[0.5]], [[1]], false);
    await t("frac", [[1]], [[0]], [[1]], false);
    await t("frac", [[2]], [[0]], [[1]], false);
});
test("frac_", async () => {
    await t("frac_", [[-2]], [[0]], [], false);
    await t("frac_", [[-1]], [[0]], [], false);
    await t("frac_", [[-0.5]], [[-0.5]], [], false);
    await t("frac_", [[0]], [[0]], [], false);
    await t("frac_", [[0.5]], [[0.5]], [], false);
    await t("frac_", [[1]], [[0]], [], false);
    await t("frac_", [[2]], [[0]], [], false);
});
test("log", async () => {
    await t("log", [[-2]], [["NaN"]], [[-0.5]], false);
    await t("log", [[-1]], [["NaN"]], [[-1]], false);
    await t("log", [[-0.5]], [["NaN"]], [[-2]], false);
    await t("log", [[0]], [["-Inf"]], [["+Inf"]], false);
    await t("log", [[0.5]], [[-0.6931471824645996]], [[2]], false);
    await t("log", [[1]], [[0]], [[1]], false);
    await t("log", [[2]], [[0.6931471824645996]], [[0.5]], false);
});
test("log_", async () => {
    await t("log_", [[-2]], [["NaN"]], [], false);
    await t("log_", [[-1]], [["NaN"]], [], false);
    await t("log_", [[-0.5]], [["NaN"]], [], false);
    await t("log_", [[0]], [["-Inf"]], [], false);
    await t("log_", [[0.5]], [[-0.6931471824645996]], [], false);
    await t("log_", [[1]], [[0]], [], false);
    await t("log_", [[2]], [[0.6931471824645996]], [], false);
});
test("log10", async () => {
    await t("log10", [[-2]], [["NaN"]], [[-0.21714723110198975]], false);
    await t("log10", [[-1]], [["NaN"]], [[-0.4342944622039795]], false);
    await t("log10", [[-0.5]], [["NaN"]], [[-0.868588924407959]], false);
    await t("log10", [[0]], [["-Inf"]], [["+Inf"]], false);
    await t("log10", [[0.5]], [[-0.3010300099849701]], [[0.868588924407959]], false);
    await t("log10", [[1]], [[0]], [[0.4342944622039795]], false);
    await t("log10", [[2]], [[0.3010300099849701]], [[0.21714723110198975]], false);
});
test("log10_", async () => {
    await t("log10_", [[-2]], [["NaN"]], [], false);
    await t("log10_", [[-1]], [["NaN"]], [], false);
    await t("log10_", [[-0.5]], [["NaN"]], [], false);
    await t("log10_", [[0]], [["-Inf"]], [], false);
    await t("log10_", [[0.5]], [[-0.3010300099849701]], [], false);
    await t("log10_", [[1]], [[0]], [], false);
    await t("log10_", [[2]], [[0.3010300099849701]], [], false);
});
test("log1p", async () => {
    await t("log1p", [[-2]], [["NaN"]], [[-1]], false);
    await t("log1p", [[-1]], [["-Inf"]], [["+Inf"]], false);
    await t("log1p", [[-0.5]], [[-0.6931471824645996]], [[2]], false);
    await t("log1p", [[0]], [[0]], [[1]], false);
    await t("log1p", [[0.5]], [[0.40546509623527527]], [[0.6666666865348816]], false);
    await t("log1p", [[1]], [[0.6931471824645996]], [[0.5]], false);
    await t("log1p", [[2]], [[1.0986123085021973]], [[0.3333333432674408]], false);
});
test("log1p_", async () => {
    await t("log1p_", [[-2]], [["NaN"]], [], false);
    await t("log1p_", [[-1]], [["-Inf"]], [], false);
    await t("log1p_", [[-0.5]], [[-0.6931471824645996]], [], false);
    await t("log1p_", [[0]], [[0]], [], false);
    await t("log1p_", [[0.5]], [[0.40546509623527527]], [], false);
    await t("log1p_", [[1]], [[0.6931471824645996]], [], false);
    await t("log1p_", [[2]], [[1.0986123085021973]], [], false);
});
test("log2", async () => {
    await t("log2", [[-2]], [["NaN"]], [[-0.7213475108146667]], false);
    await t("log2", [[-1]], [["NaN"]], [[-1.4426950216293335]], false);
    await t("log2", [[-0.5]], [["NaN"]], [[-2.885390043258667]], false);
    await t("log2", [[0]], [["-Inf"]], [["+Inf"]], false);
    await t("log2", [[0.5]], [[-1]], [[2.885390043258667]], false);
    await t("log2", [[1]], [[0]], [[1.4426950216293335]], false);
    await t("log2", [[2]], [[1]], [[0.7213475108146667]], false);
});
test("log2_", async () => {
    await t("log2_", [[-2]], [["NaN"]], [], false);
    await t("log2_", [[-1]], [["NaN"]], [], false);
    await t("log2_", [[-0.5]], [["NaN"]], [], false);
    await t("log2_", [[0]], [["-Inf"]], [], false);
    await t("log2_", [[0.5]], [[-1]], [], false);
    await t("log2_", [[1]], [[0]], [], false);
    await t("log2_", [[2]], [[1]], [], false);
});
test("neg", async () => {
    await t("neg", [[-2]], [[2]], [[-1]], false);
    await t("neg", [[-1]], [[1]], [[-1]], false);
    await t("neg", [[-0.5]], [[0.5]], [[-1]], false);
    await t("neg", [[0]], [[0]], [[-1]], false);
    await t("neg", [[0.5]], [[-0.5]], [[-1]], false);
    await t("neg", [[1]], [[-1]], [[-1]], false);
    await t("neg", [[2]], [[-2]], [[-1]], false);
});
test("neg_", async () => {
    await t("neg_", [[-2]], [[2]], [], false);
    await t("neg_", [[-1]], [[1]], [], false);
    await t("neg_", [[-0.5]], [[0.5]], [], false);
    await t("neg_", [[0]], [[0]], [], false);
    await t("neg_", [[0.5]], [[-0.5]], [], false);
    await t("neg_", [[1]], [[-1]], [], false);
    await t("neg_", [[2]], [[-2]], [], false);
});
test("positive", async () => {
    await t("positive", [[-2]], [[-2]], [[1]], false);
    await t("positive", [[-1]], [[-1]], [[1]], false);
    await t("positive", [[-0.5]], [[-0.5]], [[1]], false);
    await t("positive", [[0]], [[0]], [[1]], false);
    await t("positive", [[0.5]], [[0.5]], [[1]], false);
    await t("positive", [[1]], [[1]], [[1]], false);
    await t("positive", [[2]], [[2]], [[1]], false);
});
test("rad2deg", async () => {
    await t("rad2deg", [[-2]], [[-114.59156036376953]], [[57.295780181884766]], false);
    await t("rad2deg", [[-1]], [[-57.295780181884766]], [[57.295780181884766]], false);
    await t("rad2deg", [[-0.5]], [[-28.647890090942383]], [[57.295780181884766]], false);
    await t("rad2deg", [[0]], [[0]], [[57.295780181884766]], false);
    await t("rad2deg", [[0.5]], [[28.647890090942383]], [[57.295780181884766]], false);
    await t("rad2deg", [[1]], [[57.295780181884766]], [[57.295780181884766]], false);
    await t("rad2deg", [[2]], [[114.59156036376953]], [[57.295780181884766]], false);
});
test("rad2deg_", async () => {
    await t("rad2deg_", [[-2]], [[-114.59156036376953]], [], false);
    await t("rad2deg_", [[-1]], [[-57.295780181884766]], [], false);
    await t("rad2deg_", [[-0.5]], [[-28.647890090942383]], [], false);
    await t("rad2deg_", [[0]], [[0]], [], false);
    await t("rad2deg_", [[0.5]], [[28.647890090942383]], [], false);
    await t("rad2deg_", [[1]], [[57.295780181884766]], [], false);
    await t("rad2deg_", [[2]], [[114.59156036376953]], [], false);
});
test("reciprocal", async () => {
    await t("reciprocal", [[-2]], [[-0.5]], [[-0.25]], false);
    await t("reciprocal", [[-1]], [[-1]], [[-1]], false);
    await t("reciprocal", [[-0.5]], [[-2]], [[-4]], false);
    await t("reciprocal", [[0]], [["+Inf"]], [["-Inf"]], false);
    await t("reciprocal", [[0.5]], [[2]], [[-4]], false);
    await t("reciprocal", [[1]], [[1]], [[-1]], false);
    await t("reciprocal", [[2]], [[0.5]], [[-0.25]], false);
});
test("reciprocal_", async () => {
    await t("reciprocal_", [[-2]], [[-0.5]], [], false);
    await t("reciprocal_", [[-1]], [[-1]], [], false);
    await t("reciprocal_", [[-0.5]], [[-2]], [], false);
    await t("reciprocal_", [[0]], [["+Inf"]], [], false);
    await t("reciprocal_", [[0.5]], [[2]], [], false);
    await t("reciprocal_", [[1]], [[1]], [], false);
    await t("reciprocal_", [[2]], [[0.5]], [], false);
});
test("round", async () => {
    await t("round", [[-2]], [[-2]], [[0]], false);
    await t("round", [[-1]], [[-1]], [[0]], false);
    await t("round", [[-0.5]], [[0]], [[0]], false);
    await t("round", [[0]], [[0]], [[0]], false);
    await t("round", [[0.5]], [[0]], [[0]], false);
    await t("round", [[1]], [[1]], [[0]], false);
    await t("round", [[2]], [[2]], [[0]], false);
});
test("round_", async () => {
    await t("round_", [[-2]], [[-2]], [], false);
    await t("round_", [[-1]], [[-1]], [], false);
    await t("round_", [[-0.5]], [[0]], [], false);
    await t("round_", [[0]], [[0]], [], false);
    await t("round_", [[0.5]], [[0]], [], false);
    await t("round_", [[1]], [[1]], [], false);
    await t("round_", [[2]], [[2]], [], false);
});
test("rsqrt", async () => {
    await t("rsqrt", [[-2]], [["NaN"]], [["NaN"]], false);
    await t("rsqrt", [[-1]], [["NaN"]], [["NaN"]], false);
    await t("rsqrt", [[-0.5]], [["NaN"]], [["NaN"]], false);
    await t("rsqrt", [[0]], [["+Inf"]], [["-Inf"]], false);
    await t("rsqrt", [[0.5]], [[1.4142135381698608]], [[-1.4142134189605713]], false);
    await t("rsqrt", [[1]], [[1]], [[-0.5]], false);
    await t("rsqrt", [[2]], [[0.7071067690849304]], [[-0.1767766773700714]], false);
});
test("rsqrt_", async () => {
    await t("rsqrt_", [[-2]], [["NaN"]], [], false);
    await t("rsqrt_", [[-1]], [["NaN"]], [], false);
    await t("rsqrt_", [[-0.5]], [["NaN"]], [], false);
    await t("rsqrt_", [[0]], [["+Inf"]], [], false);
    await t("rsqrt_", [[0.5]], [[1.4142135381698608]], [], false);
    await t("rsqrt_", [[1]], [[1]], [], false);
    await t("rsqrt_", [[2]], [[0.7071067690849304]], [], false);
});
test("sigmoid", async () => {
    await t("sigmoid", [[-2]], [[0.11920291930437088]], [[0.10499358177185059]], false);
    await t("sigmoid", [[-1]], [[0.2689414322376251]], [[0.1966119408607483]], false);
    await t("sigmoid", [[-0.5]], [[0.3775406777858734]], [[0.23500370979309082]], false);
    await t("sigmoid", [[0]], [[0.5]], [[0.25]], false);
    await t("sigmoid", [[0.5]], [[0.622459352016449]], [[0.23500370979309082]], false);
    await t("sigmoid", [[1]], [[0.7310585975646973]], [[0.1966119259595871]], false);
    await t("sigmoid", [[2]], [[0.8807970285415649]], [[0.10499362647533417]], false);
});
test("sigmoid_", async () => {
    await t("sigmoid_", [[-2]], [[0.11920291930437088]], [], false);
    await t("sigmoid_", [[-1]], [[0.2689414322376251]], [], false);
    await t("sigmoid_", [[-0.5]], [[0.3775406777858734]], [], false);
    await t("sigmoid_", [[0]], [[0.5]], [], false);
    await t("sigmoid_", [[0.5]], [[0.622459352016449]], [], false);
    await t("sigmoid_", [[1]], [[0.7310585975646973]], [], false);
    await t("sigmoid_", [[2]], [[0.8807970285415649]], [], false);
});
test("sign", async () => {
    await t("sign", [[-2]], [[-1]], [[0]], false);
    await t("sign", [[-1]], [[-1]], [[0]], false);
    await t("sign", [[-0.5]], [[-1]], [[0]], false);
    await t("sign", [[0]], [[0]], [[0]], false);
    await t("sign", [[0.5]], [[1]], [[0]], false);
    await t("sign", [[1]], [[1]], [[0]], false);
    await t("sign", [[2]], [[1]], [[0]], false);
});
test("sign_", async () => {
    await t("sign_", [[-2]], [[-1]], [], false);
    await t("sign_", [[-1]], [[-1]], [], false);
    await t("sign_", [[-0.5]], [[-1]], [], false);
    await t("sign_", [[0]], [[0]], [], false);
    await t("sign_", [[0.5]], [[1]], [], false);
    await t("sign_", [[1]], [[1]], [], false);
    await t("sign_", [[2]], [[1]], [], false);
});
test("sin", async () => {
    await t("sin", [[-2]], [[-0.9092974066734314]], [[-0.416146844625473]], false);
    await t("sin", [[-1]], [[-0.8414709568023682]], [[0.5403023362159729]], false);
    await t("sin", [[-0.5]], [[-0.4794255495071411]], [[0.8775825500488281]], false);
    await t("sin", [[0]], [[0]], [[1]], false);
    await t("sin", [[0.5]], [[0.4794255495071411]], [[0.8775825500488281]], false);
    await t("sin", [[1]], [[0.8414709568023682]], [[0.5403023362159729]], false);
    await t("sin", [[2]], [[0.9092974066734314]], [[-0.416146844625473]], false);
});
test("sin_", async () => {
    await t("sin_", [[-2]], [[-0.9092974066734314]], [], false);
    await t("sin_", [[-1]], [[-0.8414709568023682]], [], false);
    await t("sin_", [[-0.5]], [[-0.4794255495071411]], [], false);
    await t("sin_", [[0]], [[0]], [], false);
    await t("sin_", [[0.5]], [[0.4794255495071411]], [], false);
    await t("sin_", [[1]], [[0.8414709568023682]], [], false);
    await t("sin_", [[2]], [[0.9092974066734314]], [], false);
});
test("sinc", async () => {
    await t("sinc", [[-2]], [[2.7827534054836178e-8]], [[-0.5]], false);
    await t("sinc", [[-1]], [[-2.7827534054836178e-8]], [[1]], false);
    await t("sinc", [[-0.5]], [[0.6366197466850281]], [[1.2732396125793457]], false);
    await t("sinc", [[0]], [[1]], [[0]], false);
    await t("sinc", [[0.5]], [[0.6366197466850281]], [[-1.2732396125793457]], false);
    await t("sinc", [[1]], [[-2.7827534054836178e-8]], [[-1]], false);
    await t("sinc", [[2]], [[2.7827534054836178e-8]], [[0.5]], false);
});
test("sinc_", async () => {
    await t("sinc_", [[-2]], [[2.7827534054836178e-8]], [], false);
    await t("sinc_", [[-1]], [[-2.7827534054836178e-8]], [], false);
    await t("sinc_", [[-0.5]], [[0.6366197466850281]], [], false);
    await t("sinc_", [[0]], [[1]], [], false);
    await t("sinc_", [[0.5]], [[0.6366197466850281]], [], false);
    await t("sinc_", [[1]], [[-2.7827534054836178e-8]], [], false);
    await t("sinc_", [[2]], [[2.7827534054836178e-8]], [], false);
});
test("sinh", async () => {
    await t("sinh", [[-2]], [[-3.6268603801727295]], [[3.762195587158203]], false);
    await t("sinh", [[-1]], [[-1.175201177597046]], [[1.5430806875228882]], false);
    await t("sinh", [[-0.5]], [[-0.5210952758789062]], [[1.1276259422302246]], false);
    await t("sinh", [[0]], [[0]], [[1]], false);
    await t("sinh", [[0.5]], [[0.5210952758789062]], [[1.1276259422302246]], false);
    await t("sinh", [[1]], [[1.175201177597046]], [[1.5430806875228882]], false);
    await t("sinh", [[2]], [[3.6268603801727295]], [[3.762195587158203]], false);
});
test("sinh_", async () => {
    await t("sinh_", [[-2]], [[-3.6268603801727295]], [], false);
    await t("sinh_", [[-1]], [[-1.175201177597046]], [], false);
    await t("sinh_", [[-0.5]], [[-0.5210952758789062]], [], false);
    await t("sinh_", [[0]], [[0]], [], false);
    await t("sinh_", [[0.5]], [[0.5210952758789062]], [], false);
    await t("sinh_", [[1]], [[1.175201177597046]], [], false);
    await t("sinh_", [[2]], [[3.6268603801727295]], [], false);
});
test("sqrt", async () => {
    await t("sqrt", [[-2]], [["NaN"]], [["NaN"]], false);
    await t("sqrt", [[-1]], [["NaN"]], [["NaN"]], false);
    await t("sqrt", [[-0.5]], [["NaN"]], [["NaN"]], false);
    await t("sqrt", [[0]], [[0]], [["+Inf"]], false);
    await t("sqrt", [[0.5]], [[0.7071067690849304]], [[0.7071067690849304]], false);
    await t("sqrt", [[1]], [[1]], [[0.5]], false);
    await t("sqrt", [[2]], [[1.4142135381698608]], [[0.3535533845424652]], false);
});
test("sqrt_", async () => {
    await t("sqrt_", [[-2]], [["NaN"]], [], false);
    await t("sqrt_", [[-1]], [["NaN"]], [], false);
    await t("sqrt_", [[-0.5]], [["NaN"]], [], false);
    await t("sqrt_", [[0]], [[0]], [], false);
    await t("sqrt_", [[0.5]], [[0.7071067690849304]], [], false);
    await t("sqrt_", [[1]], [[1]], [], false);
    await t("sqrt_", [[2]], [[1.4142135381698608]], [], false);
});
test("square", async () => {
    await t("square", [[-2]], [[4]], [[-4]], false);
    await t("square", [[-1]], [[1]], [[-2]], false);
    await t("square", [[-0.5]], [[0.25]], [[-1]], false);
    await t("square", [[0]], [[0]], [[0]], false);
    await t("square", [[0.5]], [[0.25]], [[1]], false);
    await t("square", [[1]], [[1]], [[2]], false);
    await t("square", [[2]], [[4]], [[4]], false);
});
test("square_", async () => {
    await t("square_", [[-2]], [[4]], [], false);
    await t("square_", [[-1]], [[1]], [], false);
    await t("square_", [[-0.5]], [[0.25]], [], false);
    await t("square_", [[0]], [[0]], [], false);
    await t("square_", [[0.5]], [[0.25]], [], false);
    await t("square_", [[1]], [[1]], [], false);
    await t("square_", [[2]], [[4]], [], false);
});
test("tan", async () => {
    await t("tan", [[-2]], [[2.185039758682251]], [[5.7743988037109375]], false);
    await t("tan", [[-1]], [[-1.5574077367782593]], [[3.425518751144409]], false);
    await t("tan", [[-0.5]], [[-0.5463024973869324]], [[1.2984464168548584]], false);
    await t("tan", [[0]], [[0]], [[1]], false);
    await t("tan", [[0.5]], [[0.5463024973869324]], [[1.2984464168548584]], false);
    await t("tan", [[1]], [[1.5574077367782593]], [[3.425518751144409]], false);
    await t("tan", [[2]], [[-2.185039758682251]], [[5.7743988037109375]], false);
});
test("tan_", async () => {
    await t("tan_", [[-2]], [[2.185039758682251]], [], false);
    await t("tan_", [[-1]], [[-1.5574077367782593]], [], false);
    await t("tan_", [[-0.5]], [[-0.5463024973869324]], [], false);
    await t("tan_", [[0]], [[0]], [], false);
    await t("tan_", [[0.5]], [[0.5463024973869324]], [], false);
    await t("tan_", [[1]], [[1.5574077367782593]], [], false);
    await t("tan_", [[2]], [[-2.185039758682251]], [], false);
});
test("tanh", async () => {
    await t("tanh", [[-2]], [[-0.9640275835990906]], [[0.07065081596374512]], false);
    await t("tanh", [[-1]], [[-0.7615941762924194]], [[0.41997432708740234]], false);
    await t("tanh", [[-0.5]], [[-0.46211716532707214]], [[0.7864477038383484]], false);
    await t("tanh", [[0]], [[0]], [[1]], false);
    await t("tanh", [[0.5]], [[0.46211716532707214]], [[0.7864477038383484]], false);
    await t("tanh", [[1]], [[0.7615941762924194]], [[0.41997432708740234]], false);
    await t("tanh", [[2]], [[0.9640275835990906]], [[0.07065081596374512]], false);
});
test("tanh_", async () => {
    await t("tanh_", [[-2]], [[-0.9640275835990906]], [], false);
    await t("tanh_", [[-1]], [[-0.7615941762924194]], [], false);
    await t("tanh_", [[-0.5]], [[-0.46211716532707214]], [], false);
    await t("tanh_", [[0]], [[0]], [], false);
    await t("tanh_", [[0.5]], [[0.46211716532707214]], [], false);
    await t("tanh_", [[1]], [[0.7615941762924194]], [], false);
    await t("tanh_", [[2]], [[0.9640275835990906]], [], false);
});
test("trunc", async () => {
    await t("trunc", [[-2]], [[-2]], [[0]], false);
    await t("trunc", [[-1]], [[-1]], [[0]], false);
    await t("trunc", [[-0.5]], [[0]], [[0]], false);
    await t("trunc", [[0]], [[0]], [[0]], false);
    await t("trunc", [[0.5]], [[0]], [[0]], false);
    await t("trunc", [[1]], [[1]], [[0]], false);
    await t("trunc", [[2]], [[2]], [[0]], false);
});
test("trunc_", async () => {
    await t("trunc_", [[-2]], [[-2]], [], false);
    await t("trunc_", [[-1]], [[-1]], [], false);
    await t("trunc_", [[-0.5]], [[0]], [], false);
    await t("trunc_", [[0]], [[0]], [], false);
    await t("trunc_", [[0.5]], [[0]], [], false);
    await t("trunc_", [[1]], [[1]], [], false);
    await t("trunc_", [[2]], [[2]], [], false);
});