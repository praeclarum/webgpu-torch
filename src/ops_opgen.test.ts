import { runOpgenTestForward as f, runOpgenTestBackward as b } from "./ops_opgen_test_support";
test("abs([-2])", async () => {
    await f("abs", [[-2]], [[2]]);
});
test("abs([-2]) gradient", async () => {
    await b("abs", [[-2]], [[-1]], false);
});
test("abs([-1])", async () => {
    await f("abs", [[-1]], [[1]]);
});
test("abs([-1]) gradient", async () => {
    await b("abs", [[-1]], [[-1]], false);
});
test("abs([-0.5])", async () => {
    await f("abs", [[-0.5]], [[0.5]]);
});
test("abs([-0.5]) gradient", async () => {
    await b("abs", [[-0.5]], [[-1]], false);
});
test("abs([0])", async () => {
    await f("abs", [[0]], [[0]]);
});
test("abs([0]) gradient", async () => {
    await b("abs", [[0]], [[0]], false);
});
test("abs([0.5])", async () => {
    await f("abs", [[0.5]], [[0.5]]);
});
test("abs([0.5]) gradient", async () => {
    await b("abs", [[0.5]], [[1]], false);
});
test("abs([1])", async () => {
    await f("abs", [[1]], [[1]]);
});
test("abs([1]) gradient", async () => {
    await b("abs", [[1]], [[1]], false);
});
test("abs([2])", async () => {
    await f("abs", [[2]], [[2]]);
});
test("abs([2]) gradient", async () => {
    await b("abs", [[2]], [[1]], false);
});
test("abs_([-2])", async () => {
    await f("abs_", [[-2]], [[2]]);
});
test("abs_([-1])", async () => {
    await f("abs_", [[-1]], [[1]]);
});
test("abs_([-0.5])", async () => {
    await f("abs_", [[-0.5]], [[0.5]]);
});
test("abs_([0])", async () => {
    await f("abs_", [[0]], [[0]]);
});
test("abs_([0.5])", async () => {
    await f("abs_", [[0.5]], [[0.5]]);
});
test("abs_([1])", async () => {
    await f("abs_", [[1]], [[1]]);
});
test("abs_([2])", async () => {
    await f("abs_", [[2]], [[2]]);
});
test("acos([-2])", async () => {
    await f("acos", [[-2]], [["NaN"]]);
});
test("acos([-2]) gradient", async () => {
    await b("acos", [[-2]], [["NaN"]], false);
});
test("acos([-1])", async () => {
    await f("acos", [[-1]], [[3.1415927410125732]]);
});
test("acos([-1]) gradient", async () => {
    await b("acos", [[-1]], [["-Inf"]], false);
});
test("acos([-0.5])", async () => {
    await f("acos", [[-0.5]], [[2.094395160675049]]);
});
test("acos([-0.5]) gradient", async () => {
    await b("acos", [[-0.5]], [[-1.154700517654419]], false);
});
test("acos([0])", async () => {
    await f("acos", [[0]], [[1.5707963705062866]]);
});
test("acos([0]) gradient", async () => {
    await b("acos", [[0]], [[-1]], false);
});
test("acos([0.5])", async () => {
    await f("acos", [[0.5]], [[1.0471975803375244]]);
});
test("acos([0.5]) gradient", async () => {
    await b("acos", [[0.5]], [[-1.154700517654419]], false);
});
test("acos([1])", async () => {
    await f("acos", [[1]], [[0]]);
});
test("acos([1]) gradient", async () => {
    await b("acos", [[1]], [["-Inf"]], false);
});
test("acos([2])", async () => {
    await f("acos", [[2]], [["NaN"]]);
});
test("acos([2]) gradient", async () => {
    await b("acos", [[2]], [["NaN"]], false);
});
test("acos_([-2])", async () => {
    await f("acos_", [[-2]], [["NaN"]]);
});
test("acos_([-1])", async () => {
    await f("acos_", [[-1]], [[3.1415927410125732]]);
});
test("acos_([-0.5])", async () => {
    await f("acos_", [[-0.5]], [[2.094395160675049]]);
});
test("acos_([0])", async () => {
    await f("acos_", [[0]], [[1.5707963705062866]]);
});
test("acos_([0.5])", async () => {
    await f("acos_", [[0.5]], [[1.0471975803375244]]);
});
test("acos_([1])", async () => {
    await f("acos_", [[1]], [[0]]);
});
test("acos_([2])", async () => {
    await f("acos_", [[2]], [["NaN"]]);
});
test("acosh([-2])", async () => {
    await f("acosh", [[-2]], [["NaN"]]);
});
test("acosh([-2]) gradient", async () => {
    await b("acosh", [[-2]], [[0.5773502588272095]], false);
});
test("acosh([-1])", async () => {
    await f("acosh", [[-1]], [["NaN"]]);
});
test("acosh([-1]) gradient", async () => {
    await b("acosh", [[-1]], [["+Inf"]], false);
});
test("acosh([-0.5])", async () => {
    await f("acosh", [[-0.5]], [["NaN"]]);
});
test("acosh([-0.5]) gradient", async () => {
    await b("acosh", [[-0.5]], [["NaN"]], false);
});
test("acosh([0])", async () => {
    await f("acosh", [[0]], [["NaN"]]);
});
test("acosh([0]) gradient", async () => {
    await b("acosh", [[0]], [["NaN"]], false);
});
test("acosh([0.5])", async () => {
    await f("acosh", [[0.5]], [["NaN"]]);
});
test("acosh([0.5]) gradient", async () => {
    await b("acosh", [[0.5]], [["NaN"]], false);
});
test("acosh([1])", async () => {
    await f("acosh", [[1]], [[0]]);
});
test("acosh([1]) gradient", async () => {
    await b("acosh", [[1]], [["+Inf"]], false);
});
test("acosh([2])", async () => {
    await f("acosh", [[2]], [[1.316957950592041]]);
});
test("acosh([2]) gradient", async () => {
    await b("acosh", [[2]], [[0.5773502588272095]], false);
});
test("acosh_([-2])", async () => {
    await f("acosh_", [[-2]], [["NaN"]]);
});
test("acosh_([-1])", async () => {
    await f("acosh_", [[-1]], [["NaN"]]);
});
test("acosh_([-0.5])", async () => {
    await f("acosh_", [[-0.5]], [["NaN"]]);
});
test("acosh_([0])", async () => {
    await f("acosh_", [[0]], [["NaN"]]);
});
test("acosh_([0.5])", async () => {
    await f("acosh_", [[0.5]], [["NaN"]]);
});
test("acosh_([1])", async () => {
    await f("acosh_", [[1]], [[0]]);
});
test("acosh_([2])", async () => {
    await f("acosh_", [[2]], [[1.316957950592041]]);
});
test("add([-0.5], [-0.5])", async () => {
    await f("add", [[-0.5],[-0.5]], [[-1]]);
});
test("add([-0.5], [-0.5]) gradient", async () => {
    await b("add", [[-0.5],[-0.5]], [[1],[1]], false);
});
test("add([-0.5], [0])", async () => {
    await f("add", [[-0.5],[0]], [[-0.5]]);
});
test("add([-0.5], [0]) gradient", async () => {
    await b("add", [[-0.5],[0]], [[1],[1]], false);
});
test("add([-0.5], [0.30000001192092896])", async () => {
    await f("add", [[-0.5],[0.30000001192092896]], [[-0.19999998807907104]]);
});
test("add([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("add", [[-0.5],[0.30000001192092896]], [[1],[1]], false);
});
test("add([0], [-0.5])", async () => {
    await f("add", [[0],[-0.5]], [[-0.5]]);
});
test("add([0], [-0.5]) gradient", async () => {
    await b("add", [[0],[-0.5]], [[1],[1]], false);
});
test("add([0], [0])", async () => {
    await f("add", [[0],[0]], [[0]]);
});
test("add([0], [0]) gradient", async () => {
    await b("add", [[0],[0]], [[1],[1]], false);
});
test("add([0], [0.30000001192092896])", async () => {
    await f("add", [[0],[0.30000001192092896]], [[0.30000001192092896]]);
});
test("add([0], [0.30000001192092896]) gradient", async () => {
    await b("add", [[0],[0.30000001192092896]], [[1],[1]], false);
});
test("add([0.30000001192092896], [-0.5])", async () => {
    await f("add", [[0.30000001192092896],[-0.5]], [[-0.19999998807907104]]);
});
test("add([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("add", [[0.30000001192092896],[-0.5]], [[1],[1]], false);
});
test("add([0.30000001192092896], [0])", async () => {
    await f("add", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("add([0.30000001192092896], [0]) gradient", async () => {
    await b("add", [[0.30000001192092896],[0]], [[1],[1]], false);
});
test("add([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("add", [[0.30000001192092896],[0.30000001192092896]], [[0.6000000238418579]]);
});
test("add([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("add", [[0.30000001192092896],[0.30000001192092896]], [[1],[1]], false);
});
test("add_([-0.5], [-0.5])", async () => {
    await f("add_", [[-0.5],[-0.5]], [[-1]]);
});
test("add_([-0.5], [0])", async () => {
    await f("add_", [[-0.5],[0]], [[-0.5]]);
});
test("add_([-0.5], [0.30000001192092896])", async () => {
    await f("add_", [[-0.5],[0.30000001192092896]], [[-0.19999998807907104]]);
});
test("add_([0], [-0.5])", async () => {
    await f("add_", [[0],[-0.5]], [[-0.5]]);
});
test("add_([0], [0])", async () => {
    await f("add_", [[0],[0]], [[0]]);
});
test("add_([0], [0.30000001192092896])", async () => {
    await f("add_", [[0],[0.30000001192092896]], [[0.30000001192092896]]);
});
test("add_([0.30000001192092896], [-0.5])", async () => {
    await f("add_", [[0.30000001192092896],[-0.5]], [[-0.19999998807907104]]);
});
test("add_([0.30000001192092896], [0])", async () => {
    await f("add_", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("add_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("add_", [[0.30000001192092896],[0.30000001192092896]], [[0.6000000238418579]]);
});
test("asin([-2])", async () => {
    await f("asin", [[-2]], [["NaN"]]);
});
test("asin([-2]) gradient", async () => {
    await b("asin", [[-2]], [["NaN"]], false);
});
test("asin([-1])", async () => {
    await f("asin", [[-1]], [[-1.5707963705062866]]);
});
test("asin([-1]) gradient", async () => {
    await b("asin", [[-1]], [["+Inf"]], false);
});
test("asin([-0.5])", async () => {
    await f("asin", [[-0.5]], [[-0.5235987901687622]]);
});
test("asin([-0.5]) gradient", async () => {
    await b("asin", [[-0.5]], [[1.154700517654419]], false);
});
test("asin([0])", async () => {
    await f("asin", [[0]], [[0]]);
});
test("asin([0]) gradient", async () => {
    await b("asin", [[0]], [[1]], false);
});
test("asin([0.5])", async () => {
    await f("asin", [[0.5]], [[0.5235987901687622]]);
});
test("asin([0.5]) gradient", async () => {
    await b("asin", [[0.5]], [[1.154700517654419]], false);
});
test("asin([1])", async () => {
    await f("asin", [[1]], [[1.5707963705062866]]);
});
test("asin([1]) gradient", async () => {
    await b("asin", [[1]], [["+Inf"]], false);
});
test("asin([2])", async () => {
    await f("asin", [[2]], [["NaN"]]);
});
test("asin([2]) gradient", async () => {
    await b("asin", [[2]], [["NaN"]], false);
});
test("asin_([-2])", async () => {
    await f("asin_", [[-2]], [["NaN"]]);
});
test("asin_([-1])", async () => {
    await f("asin_", [[-1]], [[-1.5707963705062866]]);
});
test("asin_([-0.5])", async () => {
    await f("asin_", [[-0.5]], [[-0.5235987901687622]]);
});
test("asin_([0])", async () => {
    await f("asin_", [[0]], [[0]]);
});
test("asin_([0.5])", async () => {
    await f("asin_", [[0.5]], [[0.5235987901687622]]);
});
test("asin_([1])", async () => {
    await f("asin_", [[1]], [[1.5707963705062866]]);
});
test("asin_([2])", async () => {
    await f("asin_", [[2]], [["NaN"]]);
});
test("asinh([-2])", async () => {
    await f("asinh", [[-2]], [[-1.4436354637145996]]);
});
test("asinh([-2]) gradient", async () => {
    await b("asinh", [[-2]], [[0.4472135901451111]], false);
});
test("asinh([-1])", async () => {
    await f("asinh", [[-1]], [[-0.8813735842704773]]);
});
test("asinh([-1]) gradient", async () => {
    await b("asinh", [[-1]], [[0.7071067690849304]], false);
});
test("asinh([-0.5])", async () => {
    await f("asinh", [[-0.5]], [[-0.4812118113040924]]);
});
test("asinh([-0.5]) gradient", async () => {
    await b("asinh", [[-0.5]], [[0.8944271802902222]], false);
});
test("asinh([0])", async () => {
    await f("asinh", [[0]], [[0]]);
});
test("asinh([0]) gradient", async () => {
    await b("asinh", [[0]], [[1]], false);
});
test("asinh([0.5])", async () => {
    await f("asinh", [[0.5]], [[0.4812118113040924]]);
});
test("asinh([0.5]) gradient", async () => {
    await b("asinh", [[0.5]], [[0.8944271802902222]], false);
});
test("asinh([1])", async () => {
    await f("asinh", [[1]], [[0.8813735842704773]]);
});
test("asinh([1]) gradient", async () => {
    await b("asinh", [[1]], [[0.7071067690849304]], false);
});
test("asinh([2])", async () => {
    await f("asinh", [[2]], [[1.4436354637145996]]);
});
test("asinh([2]) gradient", async () => {
    await b("asinh", [[2]], [[0.4472135901451111]], false);
});
test("asinh_([-2])", async () => {
    await f("asinh_", [[-2]], [[-1.4436354637145996]]);
});
test("asinh_([-1])", async () => {
    await f("asinh_", [[-1]], [[-0.8813735842704773]]);
});
test("asinh_([-0.5])", async () => {
    await f("asinh_", [[-0.5]], [[-0.4812118113040924]]);
});
test("asinh_([0])", async () => {
    await f("asinh_", [[0]], [[0]]);
});
test("asinh_([0.5])", async () => {
    await f("asinh_", [[0.5]], [[0.4812118113040924]]);
});
test("asinh_([1])", async () => {
    await f("asinh_", [[1]], [[0.8813735842704773]]);
});
test("asinh_([2])", async () => {
    await f("asinh_", [[2]], [[1.4436354637145996]]);
});
test("atan([-2])", async () => {
    await f("atan", [[-2]], [[-1.1071487665176392]]);
});
test("atan([-2]) gradient", async () => {
    await b("atan", [[-2]], [[0.20000000298023224]], false);
});
test("atan([-1])", async () => {
    await f("atan", [[-1]], [[-0.7853981256484985]]);
});
test("atan([-1]) gradient", async () => {
    await b("atan", [[-1]], [[0.5]], false);
});
test("atan([-0.5])", async () => {
    await f("atan", [[-0.5]], [[-0.46364760398864746]]);
});
test("atan([-0.5]) gradient", async () => {
    await b("atan", [[-0.5]], [[0.800000011920929]], false);
});
test("atan([0])", async () => {
    await f("atan", [[0]], [[0]]);
});
test("atan([0]) gradient", async () => {
    await b("atan", [[0]], [[1]], false);
});
test("atan([0.5])", async () => {
    await f("atan", [[0.5]], [[0.46364760398864746]]);
});
test("atan([0.5]) gradient", async () => {
    await b("atan", [[0.5]], [[0.800000011920929]], false);
});
test("atan([1])", async () => {
    await f("atan", [[1]], [[0.7853981256484985]]);
});
test("atan([1]) gradient", async () => {
    await b("atan", [[1]], [[0.5]], false);
});
test("atan([2])", async () => {
    await f("atan", [[2]], [[1.1071487665176392]]);
});
test("atan([2]) gradient", async () => {
    await b("atan", [[2]], [[0.20000000298023224]], false);
});
test("atan_([-2])", async () => {
    await f("atan_", [[-2]], [[-1.1071487665176392]]);
});
test("atan_([-1])", async () => {
    await f("atan_", [[-1]], [[-0.7853981256484985]]);
});
test("atan_([-0.5])", async () => {
    await f("atan_", [[-0.5]], [[-0.46364760398864746]]);
});
test("atan_([0])", async () => {
    await f("atan_", [[0]], [[0]]);
});
test("atan_([0.5])", async () => {
    await f("atan_", [[0.5]], [[0.46364760398864746]]);
});
test("atan_([1])", async () => {
    await f("atan_", [[1]], [[0.7853981256484985]]);
});
test("atan_([2])", async () => {
    await f("atan_", [[2]], [[1.1071487665176392]]);
});
test("atan2([-0.5], [-0.5])", async () => {
    await f("atan2", [[-0.5],[-0.5]], [[-2.356194496154785]]);
});
test("atan2([-0.5], [-0.5]) gradient", async () => {
    await b("atan2", [[-0.5],[-0.5]], [[-1],[1]], false);
});
test("atan2([-0.5], [0])", async () => {
    await f("atan2", [[-0.5],[0]], [[-1.5707963705062866]]);
});
test("atan2([-0.5], [0]) gradient", async () => {
    await b("atan2", [[-0.5],[0]], [[0],[2]], false);
});
test("atan2([-0.5], [0.30000001192092896])", async () => {
    await f("atan2", [[-0.5],[0.30000001192092896]], [[-1.0303767919540405]]);
});
test("atan2([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("atan2", [[-0.5],[0.30000001192092896]], [[0.8823529481887817],[1.470588207244873]], false);
});
test("atan2([0], [-0.5])", async () => {
    await f("atan2", [[0],[-0.5]], [[3.141592502593994]]);
});
test("atan2([0], [-0.5]) gradient", async () => {
    await b("atan2", [[0],[-0.5]], [[-2],[0]], false);
});
test("atan2([0], [0])", async () => {
    await f("atan2", [[0],[0]], [[0]]);
});
test("atan2([0], [0]) gradient", async () => {
    await b("atan2", [[0],[0]], [["NaN"],["NaN"]], false);
});
test("atan2([0], [0.30000001192092896])", async () => {
    await f("atan2", [[0],[0.30000001192092896]], [[0]]);
});
test("atan2([0], [0.30000001192092896]) gradient", async () => {
    await b("atan2", [[0],[0.30000001192092896]], [[3.3333332538604736],[0]], false);
});
test("atan2([0.30000001192092896], [-0.5])", async () => {
    await f("atan2", [[0.30000001192092896],[-0.5]], [[2.601173162460327]]);
});
test("atan2([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("atan2", [[0.30000001192092896],[-0.5]], [[-1.470588207244873],[-0.8823529481887817]], false);
});
test("atan2([0.30000001192092896], [0])", async () => {
    await f("atan2", [[0.30000001192092896],[0]], [[1.5707963705062866]]);
});
test("atan2([0.30000001192092896], [0]) gradient", async () => {
    await b("atan2", [[0.30000001192092896],[0]], [[0],[-3.3333332538604736]], false);
});
test("atan2([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("atan2", [[0.30000001192092896],[0.30000001192092896]], [[0.7853981852531433]]);
});
test("atan2([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("atan2", [[0.30000001192092896],[0.30000001192092896]], [[1.6666666269302368],[-1.6666666269302368]], false);
});
test("atan2_([-0.5], [-0.5])", async () => {
    await f("atan2_", [[-0.5],[-0.5]], [[-2.356194496154785]]);
});
test("atan2_([-0.5], [0])", async () => {
    await f("atan2_", [[-0.5],[0]], [[-1.5707963705062866]]);
});
test("atan2_([-0.5], [0.30000001192092896])", async () => {
    await f("atan2_", [[-0.5],[0.30000001192092896]], [[-1.0303767919540405]]);
});
test("atan2_([0], [-0.5])", async () => {
    await f("atan2_", [[0],[-0.5]], [[3.141592502593994]]);
});
test("atan2_([0], [0])", async () => {
    await f("atan2_", [[0],[0]], [[0]]);
});
test("atan2_([0], [0.30000001192092896])", async () => {
    await f("atan2_", [[0],[0.30000001192092896]], [[0]]);
});
test("atan2_([0.30000001192092896], [-0.5])", async () => {
    await f("atan2_", [[0.30000001192092896],[-0.5]], [[2.601173162460327]]);
});
test("atan2_([0.30000001192092896], [0])", async () => {
    await f("atan2_", [[0.30000001192092896],[0]], [[1.5707963705062866]]);
});
test("atan2_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("atan2_", [[0.30000001192092896],[0.30000001192092896]], [[0.7853981852531433]]);
});
test("ceil([-2])", async () => {
    await f("ceil", [[-2]], [[-2]]);
});
test("ceil([-2]) gradient", async () => {
    await b("ceil", [[-2]], [[0]], false);
});
test("ceil([-1])", async () => {
    await f("ceil", [[-1]], [[-1]]);
});
test("ceil([-1]) gradient", async () => {
    await b("ceil", [[-1]], [[0]], false);
});
test("ceil([-0.5])", async () => {
    await f("ceil", [[-0.5]], [[0]]);
});
test("ceil([-0.5]) gradient", async () => {
    await b("ceil", [[-0.5]], [[0]], false);
});
test("ceil([0])", async () => {
    await f("ceil", [[0]], [[0]]);
});
test("ceil([0]) gradient", async () => {
    await b("ceil", [[0]], [[0]], false);
});
test("ceil([0.5])", async () => {
    await f("ceil", [[0.5]], [[1]]);
});
test("ceil([0.5]) gradient", async () => {
    await b("ceil", [[0.5]], [[0]], false);
});
test("ceil([1])", async () => {
    await f("ceil", [[1]], [[1]]);
});
test("ceil([1]) gradient", async () => {
    await b("ceil", [[1]], [[0]], false);
});
test("ceil([2])", async () => {
    await f("ceil", [[2]], [[2]]);
});
test("ceil([2]) gradient", async () => {
    await b("ceil", [[2]], [[0]], false);
});
test("ceil_([-2])", async () => {
    await f("ceil_", [[-2]], [[-2]]);
});
test("ceil_([-1])", async () => {
    await f("ceil_", [[-1]], [[-1]]);
});
test("ceil_([-0.5])", async () => {
    await f("ceil_", [[-0.5]], [[0]]);
});
test("ceil_([0])", async () => {
    await f("ceil_", [[0]], [[0]]);
});
test("ceil_([0.5])", async () => {
    await f("ceil_", [[0.5]], [[1]]);
});
test("ceil_([1])", async () => {
    await f("ceil_", [[1]], [[1]]);
});
test("ceil_([2])", async () => {
    await f("ceil_", [[2]], [[2]]);
});
test("copysign([-0.5], [-0.5])", async () => {
    await f("copysign", [[-0.5],[-0.5]], [[-0.5]]);
});
test("copysign([-0.5], [-0.5]) gradient", async () => {
    await b("copysign", [[-0.5],[-0.5]], [[1],[0]], false);
});
test("copysign([-0.5], [0])", async () => {
    await f("copysign", [[-0.5],[0]], [[0.5]]);
});
test("copysign([-0.5], [0]) gradient", async () => {
    await b("copysign", [[-0.5],[0]], [[-1],[0]], false);
});
test("copysign([-0.5], [0.30000001192092896])", async () => {
    await f("copysign", [[-0.5],[0.30000001192092896]], [[0.5]]);
});
test("copysign([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("copysign", [[-0.5],[0.30000001192092896]], [[-1],[0]], false);
});
test("copysign([0], [-0.5])", async () => {
    await f("copysign", [[0],[-0.5]], [[0]]);
});
test("copysign([0], [-0.5]) gradient", async () => {
    await b("copysign", [[0],[-0.5]], [[0],[0]], false);
});
test("copysign([0], [0])", async () => {
    await f("copysign", [[0],[0]], [[0]]);
});
test("copysign([0], [0]) gradient", async () => {
    await b("copysign", [[0],[0]], [[0],[0]], false);
});
test("copysign([0], [0.30000001192092896])", async () => {
    await f("copysign", [[0],[0.30000001192092896]], [[0]]);
});
test("copysign([0], [0.30000001192092896]) gradient", async () => {
    await b("copysign", [[0],[0.30000001192092896]], [[0],[0]], false);
});
test("copysign([0.30000001192092896], [-0.5])", async () => {
    await f("copysign", [[0.30000001192092896],[-0.5]], [[-0.30000001192092896]]);
});
test("copysign([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("copysign", [[0.30000001192092896],[-0.5]], [[-1],[0]], false);
});
test("copysign([0.30000001192092896], [0])", async () => {
    await f("copysign", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("copysign([0.30000001192092896], [0]) gradient", async () => {
    await b("copysign", [[0.30000001192092896],[0]], [[1],[0]], false);
});
test("copysign([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("copysign", [[0.30000001192092896],[0.30000001192092896]], [[0.30000001192092896]]);
});
test("copysign([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("copysign", [[0.30000001192092896],[0.30000001192092896]], [[1],[0]], false);
});
test("copysign_([-0.5], [-0.5])", async () => {
    await f("copysign_", [[-0.5],[-0.5]], [[-0.5]]);
});
test("copysign_([-0.5], [0])", async () => {
    await f("copysign_", [[-0.5],[0]], [[0.5]]);
});
test("copysign_([-0.5], [0.30000001192092896])", async () => {
    await f("copysign_", [[-0.5],[0.30000001192092896]], [[0.5]]);
});
test("copysign_([0], [-0.5])", async () => {
    await f("copysign_", [[0],[-0.5]], [[0]]);
});
test("copysign_([0], [0])", async () => {
    await f("copysign_", [[0],[0]], [[0]]);
});
test("copysign_([0], [0.30000001192092896])", async () => {
    await f("copysign_", [[0],[0.30000001192092896]], [[0]]);
});
test("copysign_([0.30000001192092896], [-0.5])", async () => {
    await f("copysign_", [[0.30000001192092896],[-0.5]], [[-0.30000001192092896]]);
});
test("copysign_([0.30000001192092896], [0])", async () => {
    await f("copysign_", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("copysign_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("copysign_", [[0.30000001192092896],[0.30000001192092896]], [[0.30000001192092896]]);
});
test("cos([-2])", async () => {
    await f("cos", [[-2]], [[-0.416146844625473]]);
});
test("cos([-2]) gradient", async () => {
    await b("cos", [[-2]], [[0.9092974066734314]], false);
});
test("cos([-1])", async () => {
    await f("cos", [[-1]], [[0.5403023362159729]]);
});
test("cos([-1]) gradient", async () => {
    await b("cos", [[-1]], [[0.8414709568023682]], false);
});
test("cos([-0.5])", async () => {
    await f("cos", [[-0.5]], [[0.8775825500488281]]);
});
test("cos([-0.5]) gradient", async () => {
    await b("cos", [[-0.5]], [[0.4794255495071411]], false);
});
test("cos([0])", async () => {
    await f("cos", [[0]], [[1]]);
});
test("cos([0]) gradient", async () => {
    await b("cos", [[0]], [[0]], false);
});
test("cos([0.5])", async () => {
    await f("cos", [[0.5]], [[0.8775825500488281]]);
});
test("cos([0.5]) gradient", async () => {
    await b("cos", [[0.5]], [[-0.4794255495071411]], false);
});
test("cos([1])", async () => {
    await f("cos", [[1]], [[0.5403023362159729]]);
});
test("cos([1]) gradient", async () => {
    await b("cos", [[1]], [[-0.8414709568023682]], false);
});
test("cos([2])", async () => {
    await f("cos", [[2]], [[-0.416146844625473]]);
});
test("cos([2]) gradient", async () => {
    await b("cos", [[2]], [[-0.9092974066734314]], false);
});
test("cos_([-2])", async () => {
    await f("cos_", [[-2]], [[-0.416146844625473]]);
});
test("cos_([-1])", async () => {
    await f("cos_", [[-1]], [[0.5403023362159729]]);
});
test("cos_([-0.5])", async () => {
    await f("cos_", [[-0.5]], [[0.8775825500488281]]);
});
test("cos_([0])", async () => {
    await f("cos_", [[0]], [[1]]);
});
test("cos_([0.5])", async () => {
    await f("cos_", [[0.5]], [[0.8775825500488281]]);
});
test("cos_([1])", async () => {
    await f("cos_", [[1]], [[0.5403023362159729]]);
});
test("cos_([2])", async () => {
    await f("cos_", [[2]], [[-0.416146844625473]]);
});
test("cosh([-2])", async () => {
    await f("cosh", [[-2]], [[3.762195587158203]]);
});
test("cosh([-2]) gradient", async () => {
    await b("cosh", [[-2]], [[-3.6268603801727295]], false);
});
test("cosh([-1])", async () => {
    await f("cosh", [[-1]], [[1.5430806875228882]]);
});
test("cosh([-1]) gradient", async () => {
    await b("cosh", [[-1]], [[-1.175201177597046]], false);
});
test("cosh([-0.5])", async () => {
    await f("cosh", [[-0.5]], [[1.1276259422302246]]);
});
test("cosh([-0.5]) gradient", async () => {
    await b("cosh", [[-0.5]], [[-0.5210952758789062]], false);
});
test("cosh([0])", async () => {
    await f("cosh", [[0]], [[1]]);
});
test("cosh([0]) gradient", async () => {
    await b("cosh", [[0]], [[0]], false);
});
test("cosh([0.5])", async () => {
    await f("cosh", [[0.5]], [[1.1276259422302246]]);
});
test("cosh([0.5]) gradient", async () => {
    await b("cosh", [[0.5]], [[0.5210952758789062]], false);
});
test("cosh([1])", async () => {
    await f("cosh", [[1]], [[1.5430806875228882]]);
});
test("cosh([1]) gradient", async () => {
    await b("cosh", [[1]], [[1.175201177597046]], false);
});
test("cosh([2])", async () => {
    await f("cosh", [[2]], [[3.762195587158203]]);
});
test("cosh([2]) gradient", async () => {
    await b("cosh", [[2]], [[3.6268603801727295]], false);
});
test("cosh_([-2])", async () => {
    await f("cosh_", [[-2]], [[3.762195587158203]]);
});
test("cosh_([-1])", async () => {
    await f("cosh_", [[-1]], [[1.5430806875228882]]);
});
test("cosh_([-0.5])", async () => {
    await f("cosh_", [[-0.5]], [[1.1276259422302246]]);
});
test("cosh_([0])", async () => {
    await f("cosh_", [[0]], [[1]]);
});
test("cosh_([0.5])", async () => {
    await f("cosh_", [[0.5]], [[1.1276259422302246]]);
});
test("cosh_([1])", async () => {
    await f("cosh_", [[1]], [[1.5430806875228882]]);
});
test("cosh_([2])", async () => {
    await f("cosh_", [[2]], [[3.762195587158203]]);
});
test("deg2rad([-2])", async () => {
    await f("deg2rad", [[-2]], [[-0.03490658476948738]]);
});
test("deg2rad([-2]) gradient", async () => {
    await b("deg2rad", [[-2]], [[0.01745329238474369]], false);
});
test("deg2rad([-1])", async () => {
    await f("deg2rad", [[-1]], [[-0.01745329238474369]]);
});
test("deg2rad([-1]) gradient", async () => {
    await b("deg2rad", [[-1]], [[0.01745329238474369]], false);
});
test("deg2rad([-0.5])", async () => {
    await f("deg2rad", [[-0.5]], [[-0.008726646192371845]]);
});
test("deg2rad([-0.5]) gradient", async () => {
    await b("deg2rad", [[-0.5]], [[0.01745329238474369]], false);
});
test("deg2rad([0])", async () => {
    await f("deg2rad", [[0]], [[0]]);
});
test("deg2rad([0]) gradient", async () => {
    await b("deg2rad", [[0]], [[0.01745329238474369]], false);
});
test("deg2rad([0.5])", async () => {
    await f("deg2rad", [[0.5]], [[0.008726646192371845]]);
});
test("deg2rad([0.5]) gradient", async () => {
    await b("deg2rad", [[0.5]], [[0.01745329238474369]], false);
});
test("deg2rad([1])", async () => {
    await f("deg2rad", [[1]], [[0.01745329238474369]]);
});
test("deg2rad([1]) gradient", async () => {
    await b("deg2rad", [[1]], [[0.01745329238474369]], false);
});
test("deg2rad([2])", async () => {
    await f("deg2rad", [[2]], [[0.03490658476948738]]);
});
test("deg2rad([2]) gradient", async () => {
    await b("deg2rad", [[2]], [[0.01745329238474369]], false);
});
test("deg2rad_([-2])", async () => {
    await f("deg2rad_", [[-2]], [[-0.03490658476948738]]);
});
test("deg2rad_([-1])", async () => {
    await f("deg2rad_", [[-1]], [[-0.01745329238474369]]);
});
test("deg2rad_([-0.5])", async () => {
    await f("deg2rad_", [[-0.5]], [[-0.008726646192371845]]);
});
test("deg2rad_([0])", async () => {
    await f("deg2rad_", [[0]], [[0]]);
});
test("deg2rad_([0.5])", async () => {
    await f("deg2rad_", [[0.5]], [[0.008726646192371845]]);
});
test("deg2rad_([1])", async () => {
    await f("deg2rad_", [[1]], [[0.01745329238474369]]);
});
test("deg2rad_([2])", async () => {
    await f("deg2rad_", [[2]], [[0.03490658476948738]]);
});
test("div([-0.5], [-0.5])", async () => {
    await f("div", [[-0.5],[-0.5]], [[1]]);
});
test("div([-0.5], [-0.5]) gradient", async () => {
    await b("div", [[-0.5],[-0.5]], [[-2],[2]], false);
});
test("div([-0.5], [0])", async () => {
    await f("div", [[-0.5],[0]], [["-Inf"]]);
});
test("div([-0.5], [0]) gradient", async () => {
    await b("div", [[-0.5],[0]], [["+Inf"],["+Inf"]], false);
});
test("div([-0.5], [0.30000001192092896])", async () => {
    await f("div", [[-0.5],[0.30000001192092896]], [[-1.6666666269302368]]);
});
test("div([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("div", [[-0.5],[0.30000001192092896]], [[3.3333332538604736],[5.55555534362793]], false);
});
test("div([0], [-0.5])", async () => {
    await f("div", [[0],[-0.5]], [[0]]);
});
test("div([0], [-0.5]) gradient", async () => {
    await b("div", [[0],[-0.5]], [[-2],[0]], false);
});
test("div([0], [0])", async () => {
    await f("div", [[0],[0]], [["NaN"]]);
});
test("div([0], [0]) gradient", async () => {
    await b("div", [[0],[0]], [["+Inf"],["NaN"]], false);
});
test("div([0], [0.30000001192092896])", async () => {
    await f("div", [[0],[0.30000001192092896]], [[0]]);
});
test("div([0], [0.30000001192092896]) gradient", async () => {
    await b("div", [[0],[0.30000001192092896]], [[3.3333332538604736],[0]], false);
});
test("div([0.30000001192092896], [-0.5])", async () => {
    await f("div", [[0.30000001192092896],[-0.5]], [[-0.6000000238418579]]);
});
test("div([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("div", [[0.30000001192092896],[-0.5]], [[-2],[-1.2000000476837158]], false);
});
test("div([0.30000001192092896], [0])", async () => {
    await f("div", [[0.30000001192092896],[0]], [["+Inf"]]);
});
test("div([0.30000001192092896], [0]) gradient", async () => {
    await b("div", [[0.30000001192092896],[0]], [["+Inf"],["-Inf"]], false);
});
test("div([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("div", [[0.30000001192092896],[0.30000001192092896]], [[1]]);
});
test("div([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("div", [[0.30000001192092896],[0.30000001192092896]], [[3.3333332538604736],[-3.3333332538604736]], false);
});
test("div_([-0.5], [-0.5])", async () => {
    await f("div_", [[-0.5],[-0.5]], [[1]]);
});
test("div_([-0.5], [0])", async () => {
    await f("div_", [[-0.5],[0]], [["-Inf"]]);
});
test("div_([-0.5], [0.30000001192092896])", async () => {
    await f("div_", [[-0.5],[0.30000001192092896]], [[-1.6666666269302368]]);
});
test("div_([0], [-0.5])", async () => {
    await f("div_", [[0],[-0.5]], [[0]]);
});
test("div_([0], [0])", async () => {
    await f("div_", [[0],[0]], [["NaN"]]);
});
test("div_([0], [0.30000001192092896])", async () => {
    await f("div_", [[0],[0.30000001192092896]], [[0]]);
});
test("div_([0.30000001192092896], [-0.5])", async () => {
    await f("div_", [[0.30000001192092896],[-0.5]], [[-0.6000000238418579]]);
});
test("div_([0.30000001192092896], [0])", async () => {
    await f("div_", [[0.30000001192092896],[0]], [["+Inf"]]);
});
test("div_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("div_", [[0.30000001192092896],[0.30000001192092896]], [[1]]);
});
test("exp([-2])", async () => {
    await f("exp", [[-2]], [[0.1353352814912796]]);
});
test("exp([-2]) gradient", async () => {
    await b("exp", [[-2]], [[0.1353352814912796]], false);
});
test("exp([-1])", async () => {
    await f("exp", [[-1]], [[0.3678794503211975]]);
});
test("exp([-1]) gradient", async () => {
    await b("exp", [[-1]], [[0.3678794503211975]], false);
});
test("exp([-0.5])", async () => {
    await f("exp", [[-0.5]], [[0.6065306663513184]]);
});
test("exp([-0.5]) gradient", async () => {
    await b("exp", [[-0.5]], [[0.6065306663513184]], false);
});
test("exp([0])", async () => {
    await f("exp", [[0]], [[1]]);
});
test("exp([0]) gradient", async () => {
    await b("exp", [[0]], [[1]], false);
});
test("exp([0.5])", async () => {
    await f("exp", [[0.5]], [[1.6487212181091309]]);
});
test("exp([0.5]) gradient", async () => {
    await b("exp", [[0.5]], [[1.6487212181091309]], false);
});
test("exp([1])", async () => {
    await f("exp", [[1]], [[2.7182817459106445]]);
});
test("exp([1]) gradient", async () => {
    await b("exp", [[1]], [[2.7182817459106445]], false);
});
test("exp([2])", async () => {
    await f("exp", [[2]], [[7.389056205749512]]);
});
test("exp([2]) gradient", async () => {
    await b("exp", [[2]], [[7.389056205749512]], false);
});
test("exp_([-2])", async () => {
    await f("exp_", [[-2]], [[0.1353352814912796]]);
});
test("exp_([-1])", async () => {
    await f("exp_", [[-1]], [[0.3678794503211975]]);
});
test("exp_([-0.5])", async () => {
    await f("exp_", [[-0.5]], [[0.6065306663513184]]);
});
test("exp_([0])", async () => {
    await f("exp_", [[0]], [[1]]);
});
test("exp_([0.5])", async () => {
    await f("exp_", [[0.5]], [[1.6487212181091309]]);
});
test("exp_([1])", async () => {
    await f("exp_", [[1]], [[2.7182817459106445]]);
});
test("exp_([2])", async () => {
    await f("exp_", [[2]], [[7.389056205749512]]);
});
test("exp2([-2])", async () => {
    await f("exp2", [[-2]], [[0.25]]);
});
test("exp2([-2]) gradient", async () => {
    await b("exp2", [[-2]], [[0.1732867956161499]], false);
});
test("exp2([-1])", async () => {
    await f("exp2", [[-1]], [[0.5]]);
});
test("exp2([-1]) gradient", async () => {
    await b("exp2", [[-1]], [[0.3465735912322998]], false);
});
test("exp2([-0.5])", async () => {
    await f("exp2", [[-0.5]], [[0.7071067690849304]]);
});
test("exp2([-0.5]) gradient", async () => {
    await b("exp2", [[-0.5]], [[0.4901290535926819]], false);
});
test("exp2([0])", async () => {
    await f("exp2", [[0]], [[1]]);
});
test("exp2([0]) gradient", async () => {
    await b("exp2", [[0]], [[0.6931471824645996]], false);
});
test("exp2([0.5])", async () => {
    await f("exp2", [[0.5]], [[1.4142135381698608]]);
});
test("exp2([0.5]) gradient", async () => {
    await b("exp2", [[0.5]], [[0.9802581071853638]], false);
});
test("exp2([1])", async () => {
    await f("exp2", [[1]], [[2]]);
});
test("exp2([1]) gradient", async () => {
    await b("exp2", [[1]], [[1.3862943649291992]], false);
});
test("exp2([2])", async () => {
    await f("exp2", [[2]], [[4]]);
});
test("exp2([2]) gradient", async () => {
    await b("exp2", [[2]], [[2.7725887298583984]], false);
});
test("exp2_([-2])", async () => {
    await f("exp2_", [[-2]], [[0.25]]);
});
test("exp2_([-1])", async () => {
    await f("exp2_", [[-1]], [[0.5]]);
});
test("exp2_([-0.5])", async () => {
    await f("exp2_", [[-0.5]], [[0.7071067690849304]]);
});
test("exp2_([0])", async () => {
    await f("exp2_", [[0]], [[1]]);
});
test("exp2_([0.5])", async () => {
    await f("exp2_", [[0.5]], [[1.4142135381698608]]);
});
test("exp2_([1])", async () => {
    await f("exp2_", [[1]], [[2]]);
});
test("exp2_([2])", async () => {
    await f("exp2_", [[2]], [[4]]);
});
test("expm1([-2])", async () => {
    await f("expm1", [[-2]], [[-0.8646647334098816]]);
});
test("expm1([-2]) gradient", async () => {
    await b("expm1", [[-2]], [[0.1353352665901184]], false);
});
test("expm1([-1])", async () => {
    await f("expm1", [[-1]], [[-0.6321205496788025]]);
});
test("expm1([-1]) gradient", async () => {
    await b("expm1", [[-1]], [[0.3678794503211975]], false);
});
test("expm1([-0.5])", async () => {
    await f("expm1", [[-0.5]], [[-0.39346933364868164]]);
});
test("expm1([-0.5]) gradient", async () => {
    await b("expm1", [[-0.5]], [[0.6065306663513184]], false);
});
test("expm1([0])", async () => {
    await f("expm1", [[0]], [[0]]);
});
test("expm1([0]) gradient", async () => {
    await b("expm1", [[0]], [[1]], false);
});
test("expm1([0.5])", async () => {
    await f("expm1", [[0.5]], [[0.6487212777137756]]);
});
test("expm1([0.5]) gradient", async () => {
    await b("expm1", [[0.5]], [[1.6487212181091309]], false);
});
test("expm1([1])", async () => {
    await f("expm1", [[1]], [[1.718281865119934]]);
});
test("expm1([1]) gradient", async () => {
    await b("expm1", [[1]], [[2.7182817459106445]], false);
});
test("expm1([2])", async () => {
    await f("expm1", [[2]], [[6.389056205749512]]);
});
test("expm1([2]) gradient", async () => {
    await b("expm1", [[2]], [[7.389056205749512]], false);
});
test("expm1_([-2])", async () => {
    await f("expm1_", [[-2]], [[-0.8646647334098816]]);
});
test("expm1_([-1])", async () => {
    await f("expm1_", [[-1]], [[-0.6321205496788025]]);
});
test("expm1_([-0.5])", async () => {
    await f("expm1_", [[-0.5]], [[-0.39346933364868164]]);
});
test("expm1_([0])", async () => {
    await f("expm1_", [[0]], [[0]]);
});
test("expm1_([0.5])", async () => {
    await f("expm1_", [[0.5]], [[0.6487212777137756]]);
});
test("expm1_([1])", async () => {
    await f("expm1_", [[1]], [[1.718281865119934]]);
});
test("expm1_([2])", async () => {
    await f("expm1_", [[2]], [[6.389056205749512]]);
});
test("floor([-2])", async () => {
    await f("floor", [[-2]], [[-2]]);
});
test("floor([-2]) gradient", async () => {
    await b("floor", [[-2]], [[0]], false);
});
test("floor([-1])", async () => {
    await f("floor", [[-1]], [[-1]]);
});
test("floor([-1]) gradient", async () => {
    await b("floor", [[-1]], [[0]], false);
});
test("floor([-0.5])", async () => {
    await f("floor", [[-0.5]], [[-1]]);
});
test("floor([-0.5]) gradient", async () => {
    await b("floor", [[-0.5]], [[0]], false);
});
test("floor([0])", async () => {
    await f("floor", [[0]], [[0]]);
});
test("floor([0]) gradient", async () => {
    await b("floor", [[0]], [[0]], false);
});
test("floor([0.5])", async () => {
    await f("floor", [[0.5]], [[0]]);
});
test("floor([0.5]) gradient", async () => {
    await b("floor", [[0.5]], [[0]], false);
});
test("floor([1])", async () => {
    await f("floor", [[1]], [[1]]);
});
test("floor([1]) gradient", async () => {
    await b("floor", [[1]], [[0]], false);
});
test("floor([2])", async () => {
    await f("floor", [[2]], [[2]]);
});
test("floor([2]) gradient", async () => {
    await b("floor", [[2]], [[0]], false);
});
test("floor_([-2])", async () => {
    await f("floor_", [[-2]], [[-2]]);
});
test("floor_([-1])", async () => {
    await f("floor_", [[-1]], [[-1]]);
});
test("floor_([-0.5])", async () => {
    await f("floor_", [[-0.5]], [[-1]]);
});
test("floor_([0])", async () => {
    await f("floor_", [[0]], [[0]]);
});
test("floor_([0.5])", async () => {
    await f("floor_", [[0.5]], [[0]]);
});
test("floor_([1])", async () => {
    await f("floor_", [[1]], [[1]]);
});
test("floor_([2])", async () => {
    await f("floor_", [[2]], [[2]]);
});
test("frac([-2])", async () => {
    await f("frac", [[-2]], [[0]]);
});
test("frac([-2]) gradient", async () => {
    await b("frac", [[-2]], [[1]], false);
});
test("frac([-1])", async () => {
    await f("frac", [[-1]], [[0]]);
});
test("frac([-1]) gradient", async () => {
    await b("frac", [[-1]], [[1]], false);
});
test("frac([-0.5])", async () => {
    await f("frac", [[-0.5]], [[-0.5]]);
});
test("frac([-0.5]) gradient", async () => {
    await b("frac", [[-0.5]], [[1]], false);
});
test("frac([0])", async () => {
    await f("frac", [[0]], [[0]]);
});
test("frac([0]) gradient", async () => {
    await b("frac", [[0]], [[1]], false);
});
test("frac([0.5])", async () => {
    await f("frac", [[0.5]], [[0.5]]);
});
test("frac([0.5]) gradient", async () => {
    await b("frac", [[0.5]], [[1]], false);
});
test("frac([1])", async () => {
    await f("frac", [[1]], [[0]]);
});
test("frac([1]) gradient", async () => {
    await b("frac", [[1]], [[1]], false);
});
test("frac([2])", async () => {
    await f("frac", [[2]], [[0]]);
});
test("frac([2]) gradient", async () => {
    await b("frac", [[2]], [[1]], false);
});
test("frac_([-2])", async () => {
    await f("frac_", [[-2]], [[0]]);
});
test("frac_([-1])", async () => {
    await f("frac_", [[-1]], [[0]]);
});
test("frac_([-0.5])", async () => {
    await f("frac_", [[-0.5]], [[-0.5]]);
});
test("frac_([0])", async () => {
    await f("frac_", [[0]], [[0]]);
});
test("frac_([0.5])", async () => {
    await f("frac_", [[0.5]], [[0.5]]);
});
test("frac_([1])", async () => {
    await f("frac_", [[1]], [[0]]);
});
test("frac_([2])", async () => {
    await f("frac_", [[2]], [[0]]);
});
test("hypot([-0.5], [-0.5])", async () => {
    await f("hypot", [[-0.5],[-0.5]], [[0.7071067690849304]]);
});
test("hypot([-0.5], [-0.5]) gradient", async () => {
    await b("hypot", [[-0.5],[-0.5]], [[-0.7071067690849304],[-0.7071067690849304]], false);
});
test("hypot([-0.5], [0])", async () => {
    await f("hypot", [[-0.5],[0]], [[0.5]]);
});
test("hypot([-0.5], [0]) gradient", async () => {
    await b("hypot", [[-0.5],[0]], [[-1],[0]], false);
});
test("hypot([-0.5], [0.30000001192092896])", async () => {
    await f("hypot", [[-0.5],[0.30000001192092896]], [[0.5830951929092407]]);
});
test("hypot([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("hypot", [[-0.5],[0.30000001192092896]], [[-0.8574929237365723],[0.5144957900047302]], false);
});
test("hypot([0], [-0.5])", async () => {
    await f("hypot", [[0],[-0.5]], [[0.5]]);
});
test("hypot([0], [-0.5]) gradient", async () => {
    await b("hypot", [[0],[-0.5]], [[0],[-1]], false);
});
test("hypot([0], [0])", async () => {
    await f("hypot", [[0],[0]], [[0]]);
});
test("hypot([0], [0]) gradient", async () => {
    await b("hypot", [[0],[0]], [["NaN"],["NaN"]], false);
});
test("hypot([0], [0.30000001192092896])", async () => {
    await f("hypot", [[0],[0.30000001192092896]], [[0.30000001192092896]]);
});
test("hypot([0], [0.30000001192092896]) gradient", async () => {
    await b("hypot", [[0],[0.30000001192092896]], [[0],[1]], false);
});
test("hypot([0.30000001192092896], [-0.5])", async () => {
    await f("hypot", [[0.30000001192092896],[-0.5]], [[0.5830951929092407]]);
});
test("hypot([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("hypot", [[0.30000001192092896],[-0.5]], [[0.5144957900047302],[-0.8574929237365723]], false);
});
test("hypot([0.30000001192092896], [0])", async () => {
    await f("hypot", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("hypot([0.30000001192092896], [0]) gradient", async () => {
    await b("hypot", [[0.30000001192092896],[0]], [[1],[0]], false);
});
test("hypot([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("hypot", [[0.30000001192092896],[0.30000001192092896]], [[0.4242640733718872]]);
});
test("hypot([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("hypot", [[0.30000001192092896],[0.30000001192092896]], [[0.7071068286895752],[0.7071068286895752]], false);
});
test("hypot_([-0.5], [-0.5])", async () => {
    await f("hypot_", [[-0.5],[-0.5]], [[0.7071067690849304]]);
});
test("hypot_([-0.5], [0])", async () => {
    await f("hypot_", [[-0.5],[0]], [[0.5]]);
});
test("hypot_([-0.5], [0.30000001192092896])", async () => {
    await f("hypot_", [[-0.5],[0.30000001192092896]], [[0.5830951929092407]]);
});
test("hypot_([0], [-0.5])", async () => {
    await f("hypot_", [[0],[-0.5]], [[0.5]]);
});
test("hypot_([0], [0])", async () => {
    await f("hypot_", [[0],[0]], [[0]]);
});
test("hypot_([0], [0.30000001192092896])", async () => {
    await f("hypot_", [[0],[0.30000001192092896]], [[0.30000001192092896]]);
});
test("hypot_([0.30000001192092896], [-0.5])", async () => {
    await f("hypot_", [[0.30000001192092896],[-0.5]], [[0.5830951929092407]]);
});
test("hypot_([0.30000001192092896], [0])", async () => {
    await f("hypot_", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("hypot_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("hypot_", [[0.30000001192092896],[0.30000001192092896]], [[0.4242640733718872]]);
});
test("ldexp([-0.5], [-0.5])", async () => {
    await f("ldexp", [[-0.5],[-0.5]], [[-0.3535533845424652]]);
});
test("ldexp([-0.5], [-0.5]) gradient", async () => {
    await b("ldexp", [[-0.5],[-0.5]], [[0.7071067690849304],[-0.24506452679634094]], false);
});
test("ldexp([-0.5], [0])", async () => {
    await f("ldexp", [[-0.5],[0]], [[-0.5]]);
});
test("ldexp([-0.5], [0]) gradient", async () => {
    await b("ldexp", [[-0.5],[0]], [[1],[-0.3465735912322998]], false);
});
test("ldexp([-0.5], [0.30000001192092896])", async () => {
    await f("ldexp", [[-0.5],[0.30000001192092896]], [[-0.6155722141265869]]);
});
test("ldexp([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("ldexp", [[-0.5],[0.30000001192092896]], [[1.2311444282531738],[-0.42668214440345764]], false);
});
test("ldexp([0], [-0.5])", async () => {
    await f("ldexp", [[0],[-0.5]], [[0]]);
});
test("ldexp([0], [-0.5]) gradient", async () => {
    await b("ldexp", [[0],[-0.5]], [[0.7071067690849304],[0]], false);
});
test("ldexp([0], [0])", async () => {
    await f("ldexp", [[0],[0]], [[0]]);
});
test("ldexp([0], [0]) gradient", async () => {
    await b("ldexp", [[0],[0]], [[1],[0]], false);
});
test("ldexp([0], [0.30000001192092896])", async () => {
    await f("ldexp", [[0],[0.30000001192092896]], [[0]]);
});
test("ldexp([0], [0.30000001192092896]) gradient", async () => {
    await b("ldexp", [[0],[0.30000001192092896]], [[1.2311444282531738],[0]], false);
});
test("ldexp([0.30000001192092896], [-0.5])", async () => {
    await f("ldexp", [[0.30000001192092896],[-0.5]], [[0.2121320366859436]]);
});
test("ldexp([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("ldexp", [[0.30000001192092896],[-0.5]], [[0.7071067690849304],[0.14703872799873352]], false);
});
test("ldexp([0.30000001192092896], [0])", async () => {
    await f("ldexp", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("ldexp([0.30000001192092896], [0]) gradient", async () => {
    await b("ldexp", [[0.30000001192092896],[0]], [[1],[0.20794416964054108]], false);
});
test("ldexp([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("ldexp", [[0.30000001192092896],[0.30000001192092896]], [[0.3693433403968811]]);
});
test("ldexp([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("ldexp", [[0.30000001192092896],[0.30000001192092896]], [[1.2311444282531738],[0.2560093104839325]], false);
});
test("ldexp_([-0.5], [-0.5])", async () => {
    await f("ldexp_", [[-0.5],[-0.5]], [[-0.3535533845424652]]);
});
test("ldexp_([-0.5], [0])", async () => {
    await f("ldexp_", [[-0.5],[0]], [[-0.5]]);
});
test("ldexp_([-0.5], [0.30000001192092896])", async () => {
    await f("ldexp_", [[-0.5],[0.30000001192092896]], [[-0.6155722141265869]]);
});
test("ldexp_([0], [-0.5])", async () => {
    await f("ldexp_", [[0],[-0.5]], [[0]]);
});
test("ldexp_([0], [0])", async () => {
    await f("ldexp_", [[0],[0]], [[0]]);
});
test("ldexp_([0], [0.30000001192092896])", async () => {
    await f("ldexp_", [[0],[0.30000001192092896]], [[0]]);
});
test("ldexp_([0.30000001192092896], [-0.5])", async () => {
    await f("ldexp_", [[0.30000001192092896],[-0.5]], [[0.2121320366859436]]);
});
test("ldexp_([0.30000001192092896], [0])", async () => {
    await f("ldexp_", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("ldexp_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("ldexp_", [[0.30000001192092896],[0.30000001192092896]], [[0.3693433403968811]]);
});
test("log([-2])", async () => {
    await f("log", [[-2]], [["NaN"]]);
});
test("log([-2]) gradient", async () => {
    await b("log", [[-2]], [[-0.5]], false);
});
test("log([-1])", async () => {
    await f("log", [[-1]], [["NaN"]]);
});
test("log([-1]) gradient", async () => {
    await b("log", [[-1]], [[-1]], false);
});
test("log([-0.5])", async () => {
    await f("log", [[-0.5]], [["NaN"]]);
});
test("log([-0.5]) gradient", async () => {
    await b("log", [[-0.5]], [[-2]], false);
});
test("log([0])", async () => {
    await f("log", [[0]], [["-Inf"]]);
});
test("log([0]) gradient", async () => {
    await b("log", [[0]], [["+Inf"]], false);
});
test("log([0.5])", async () => {
    await f("log", [[0.5]], [[-0.6931471824645996]]);
});
test("log([0.5]) gradient", async () => {
    await b("log", [[0.5]], [[2]], false);
});
test("log([1])", async () => {
    await f("log", [[1]], [[0]]);
});
test("log([1]) gradient", async () => {
    await b("log", [[1]], [[1]], false);
});
test("log([2])", async () => {
    await f("log", [[2]], [[0.6931471824645996]]);
});
test("log([2]) gradient", async () => {
    await b("log", [[2]], [[0.5]], false);
});
test("log_([-2])", async () => {
    await f("log_", [[-2]], [["NaN"]]);
});
test("log_([-1])", async () => {
    await f("log_", [[-1]], [["NaN"]]);
});
test("log_([-0.5])", async () => {
    await f("log_", [[-0.5]], [["NaN"]]);
});
test("log_([0])", async () => {
    await f("log_", [[0]], [["-Inf"]]);
});
test("log_([0.5])", async () => {
    await f("log_", [[0.5]], [[-0.6931471824645996]]);
});
test("log_([1])", async () => {
    await f("log_", [[1]], [[0]]);
});
test("log_([2])", async () => {
    await f("log_", [[2]], [[0.6931471824645996]]);
});
test("log10([-2])", async () => {
    await f("log10", [[-2]], [["NaN"]]);
});
test("log10([-2]) gradient", async () => {
    await b("log10", [[-2]], [[-0.21714723110198975]], false);
});
test("log10([-1])", async () => {
    await f("log10", [[-1]], [["NaN"]]);
});
test("log10([-1]) gradient", async () => {
    await b("log10", [[-1]], [[-0.4342944622039795]], false);
});
test("log10([-0.5])", async () => {
    await f("log10", [[-0.5]], [["NaN"]]);
});
test("log10([-0.5]) gradient", async () => {
    await b("log10", [[-0.5]], [[-0.868588924407959]], false);
});
test("log10([0])", async () => {
    await f("log10", [[0]], [["-Inf"]]);
});
test("log10([0]) gradient", async () => {
    await b("log10", [[0]], [["+Inf"]], false);
});
test("log10([0.5])", async () => {
    await f("log10", [[0.5]], [[-0.3010300099849701]]);
});
test("log10([0.5]) gradient", async () => {
    await b("log10", [[0.5]], [[0.868588924407959]], false);
});
test("log10([1])", async () => {
    await f("log10", [[1]], [[0]]);
});
test("log10([1]) gradient", async () => {
    await b("log10", [[1]], [[0.4342944622039795]], false);
});
test("log10([2])", async () => {
    await f("log10", [[2]], [[0.3010300099849701]]);
});
test("log10([2]) gradient", async () => {
    await b("log10", [[2]], [[0.21714723110198975]], false);
});
test("log10_([-2])", async () => {
    await f("log10_", [[-2]], [["NaN"]]);
});
test("log10_([-1])", async () => {
    await f("log10_", [[-1]], [["NaN"]]);
});
test("log10_([-0.5])", async () => {
    await f("log10_", [[-0.5]], [["NaN"]]);
});
test("log10_([0])", async () => {
    await f("log10_", [[0]], [["-Inf"]]);
});
test("log10_([0.5])", async () => {
    await f("log10_", [[0.5]], [[-0.3010300099849701]]);
});
test("log10_([1])", async () => {
    await f("log10_", [[1]], [[0]]);
});
test("log10_([2])", async () => {
    await f("log10_", [[2]], [[0.3010300099849701]]);
});
test("log1p([-2])", async () => {
    await f("log1p", [[-2]], [["NaN"]]);
});
test("log1p([-2]) gradient", async () => {
    await b("log1p", [[-2]], [[-1]], false);
});
test("log1p([-1])", async () => {
    await f("log1p", [[-1]], [["-Inf"]]);
});
test("log1p([-1]) gradient", async () => {
    await b("log1p", [[-1]], [["+Inf"]], false);
});
test("log1p([-0.5])", async () => {
    await f("log1p", [[-0.5]], [[-0.6931471824645996]]);
});
test("log1p([-0.5]) gradient", async () => {
    await b("log1p", [[-0.5]], [[2]], false);
});
test("log1p([0])", async () => {
    await f("log1p", [[0]], [[0]]);
});
test("log1p([0]) gradient", async () => {
    await b("log1p", [[0]], [[1]], false);
});
test("log1p([0.5])", async () => {
    await f("log1p", [[0.5]], [[0.40546509623527527]]);
});
test("log1p([0.5]) gradient", async () => {
    await b("log1p", [[0.5]], [[0.6666666865348816]], false);
});
test("log1p([1])", async () => {
    await f("log1p", [[1]], [[0.6931471824645996]]);
});
test("log1p([1]) gradient", async () => {
    await b("log1p", [[1]], [[0.5]], false);
});
test("log1p([2])", async () => {
    await f("log1p", [[2]], [[1.0986123085021973]]);
});
test("log1p([2]) gradient", async () => {
    await b("log1p", [[2]], [[0.3333333432674408]], false);
});
test("log1p_([-2])", async () => {
    await f("log1p_", [[-2]], [["NaN"]]);
});
test("log1p_([-1])", async () => {
    await f("log1p_", [[-1]], [["-Inf"]]);
});
test("log1p_([-0.5])", async () => {
    await f("log1p_", [[-0.5]], [[-0.6931471824645996]]);
});
test("log1p_([0])", async () => {
    await f("log1p_", [[0]], [[0]]);
});
test("log1p_([0.5])", async () => {
    await f("log1p_", [[0.5]], [[0.40546509623527527]]);
});
test("log1p_([1])", async () => {
    await f("log1p_", [[1]], [[0.6931471824645996]]);
});
test("log1p_([2])", async () => {
    await f("log1p_", [[2]], [[1.0986123085021973]]);
});
test("log2([-2])", async () => {
    await f("log2", [[-2]], [["NaN"]]);
});
test("log2([-2]) gradient", async () => {
    await b("log2", [[-2]], [[-0.7213475108146667]], false);
});
test("log2([-1])", async () => {
    await f("log2", [[-1]], [["NaN"]]);
});
test("log2([-1]) gradient", async () => {
    await b("log2", [[-1]], [[-1.4426950216293335]], false);
});
test("log2([-0.5])", async () => {
    await f("log2", [[-0.5]], [["NaN"]]);
});
test("log2([-0.5]) gradient", async () => {
    await b("log2", [[-0.5]], [[-2.885390043258667]], false);
});
test("log2([0])", async () => {
    await f("log2", [[0]], [["-Inf"]]);
});
test("log2([0]) gradient", async () => {
    await b("log2", [[0]], [["+Inf"]], false);
});
test("log2([0.5])", async () => {
    await f("log2", [[0.5]], [[-1]]);
});
test("log2([0.5]) gradient", async () => {
    await b("log2", [[0.5]], [[2.885390043258667]], false);
});
test("log2([1])", async () => {
    await f("log2", [[1]], [[0]]);
});
test("log2([1]) gradient", async () => {
    await b("log2", [[1]], [[1.4426950216293335]], false);
});
test("log2([2])", async () => {
    await f("log2", [[2]], [[1]]);
});
test("log2([2]) gradient", async () => {
    await b("log2", [[2]], [[0.7213475108146667]], false);
});
test("log2_([-2])", async () => {
    await f("log2_", [[-2]], [["NaN"]]);
});
test("log2_([-1])", async () => {
    await f("log2_", [[-1]], [["NaN"]]);
});
test("log2_([-0.5])", async () => {
    await f("log2_", [[-0.5]], [["NaN"]]);
});
test("log2_([0])", async () => {
    await f("log2_", [[0]], [["-Inf"]]);
});
test("log2_([0.5])", async () => {
    await f("log2_", [[0.5]], [[-1]]);
});
test("log2_([1])", async () => {
    await f("log2_", [[1]], [[0]]);
});
test("log2_([2])", async () => {
    await f("log2_", [[2]], [[1]]);
});
test("logaddexp([-0.5], [-0.5])", async () => {
    await f("logaddexp", [[-0.5],[-0.5]], [[0.1931471824645996]]);
});
test("logaddexp([-0.5], [-0.5]) gradient", async () => {
    await b("logaddexp", [[-0.5],[-0.5]], [[0.5],[0.5]], false);
});
test("logaddexp([-0.5], [0])", async () => {
    await f("logaddexp", [[-0.5],[0]], [[0.4740769863128662]]);
});
test("logaddexp([-0.5], [0]) gradient", async () => {
    await b("logaddexp", [[-0.5],[0]], [[0.3775406777858734],[0.622459352016449]], false);
});
test("logaddexp([-0.5], [0.30000001192092896])", async () => {
    await f("logaddexp", [[-0.5],[0.30000001192092896]], [[0.6711006164550781]]);
});
test("logaddexp([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("logaddexp", [[-0.5],[0.30000001192092896]], [[0.31002551317214966],[0.6899744868278503]], false);
});
test("logaddexp([0], [-0.5])", async () => {
    await f("logaddexp", [[0],[-0.5]], [[0.4740769863128662]]);
});
test("logaddexp([0], [-0.5]) gradient", async () => {
    await b("logaddexp", [[0],[-0.5]], [[0.622459352016449],[0.3775406777858734]], false);
});
test("logaddexp([0], [0])", async () => {
    await f("logaddexp", [[0],[0]], [[0.6931471824645996]]);
});
test("logaddexp([0], [0]) gradient", async () => {
    await b("logaddexp", [[0],[0]], [[0.5],[0.5]], false);
});
test("logaddexp([0], [0.30000001192092896])", async () => {
    await f("logaddexp", [[0],[0.30000001192092896]], [[0.8543552756309509]]);
});
test("logaddexp([0], [0.30000001192092896]) gradient", async () => {
    await b("logaddexp", [[0],[0.30000001192092896]], [[0.4255574941635132],[0.5744425058364868]], false);
});
test("logaddexp([0.30000001192092896], [-0.5])", async () => {
    await f("logaddexp", [[0.30000001192092896],[-0.5]], [[0.6711006164550781]]);
});
test("logaddexp([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("logaddexp", [[0.30000001192092896],[-0.5]], [[0.6899744868278503],[0.31002551317214966]], false);
});
test("logaddexp([0.30000001192092896], [0])", async () => {
    await f("logaddexp", [[0.30000001192092896],[0]], [[0.8543552756309509]]);
});
test("logaddexp([0.30000001192092896], [0]) gradient", async () => {
    await b("logaddexp", [[0.30000001192092896],[0]], [[0.5744425058364868],[0.4255574941635132]], false);
});
test("logaddexp([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("logaddexp", [[0.30000001192092896],[0.30000001192092896]], [[0.9931471943855286]]);
});
test("logaddexp([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("logaddexp", [[0.30000001192092896],[0.30000001192092896]], [[0.5],[0.5]], false);
});
test("logaddexp2([-0.5], [-0.5])", async () => {
    await f("logaddexp2", [[-0.5],[-0.5]], [[0.5]]);
});
test("logaddexp2([-0.5], [-0.5]) gradient", async () => {
    await b("logaddexp2", [[-0.5],[-0.5]], [[0.5],[0.5]], false);
});
test("logaddexp2([-0.5], [0])", async () => {
    await f("logaddexp2", [[-0.5],[0]], [[0.7715533375740051]]);
});
test("logaddexp2([-0.5], [0]) gradient", async () => {
    await b("logaddexp2", [[-0.5],[0]], [[0.41421353816986084],[0.5857864022254944]], false);
});
test("logaddexp2([-0.5], [0.30000001192092896])", async () => {
    await f("logaddexp2", [[-0.5],[0.30000001192092896]], [[0.9547555446624756]]);
});
test("logaddexp2([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("logaddexp2", [[-0.5],[0.30000001192092896]], [[0.3648168742656708],[0.6351830959320068]], false);
});
test("logaddexp2([0], [-0.5])", async () => {
    await f("logaddexp2", [[0],[-0.5]], [[0.7715533375740051]]);
});
test("logaddexp2([0], [-0.5]) gradient", async () => {
    await b("logaddexp2", [[0],[-0.5]], [[0.5857864022254944],[0.41421353816986084]], false);
});
test("logaddexp2([0], [0])", async () => {
    await f("logaddexp2", [[0],[0]], [[1]]);
});
test("logaddexp2([0], [0]) gradient", async () => {
    await b("logaddexp2", [[0],[0]], [[0.5],[0.5]], false);
});
test("logaddexp2([0], [0.30000001192092896])", async () => {
    await f("logaddexp2", [[0],[0.30000001192092896]], [[1.1577839851379395]]);
});
test("logaddexp2([0], [0.30000001192092896]) gradient", async () => {
    await b("logaddexp2", [[0],[0.30000001192092896]], [[0.4482004642486572],[0.5517995357513428]], false);
});
test("logaddexp2([0.30000001192092896], [-0.5])", async () => {
    await f("logaddexp2", [[0.30000001192092896],[-0.5]], [[0.9547555446624756]]);
});
test("logaddexp2([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("logaddexp2", [[0.30000001192092896],[-0.5]], [[0.6351830959320068],[0.3648168742656708]], false);
});
test("logaddexp2([0.30000001192092896], [0])", async () => {
    await f("logaddexp2", [[0.30000001192092896],[0]], [[1.1577839851379395]]);
});
test("logaddexp2([0.30000001192092896], [0]) gradient", async () => {
    await b("logaddexp2", [[0.30000001192092896],[0]], [[0.5517995357513428],[0.4482004642486572]], false);
});
test("logaddexp2([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("logaddexp2", [[0.30000001192092896],[0.30000001192092896]], [[1.2999999523162842]]);
});
test("logaddexp2([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("logaddexp2", [[0.30000001192092896],[0.30000001192092896]], [[0.5],[0.5]], false);
});
test("mul([-0.5], [-0.5])", async () => {
    await f("mul", [[-0.5],[-0.5]], [[0.25]]);
});
test("mul([-0.5], [-0.5]) gradient", async () => {
    await b("mul", [[-0.5],[-0.5]], [[-0.5],[-0.5]], false);
});
test("mul([-0.5], [0])", async () => {
    await f("mul", [[-0.5],[0]], [[0]]);
});
test("mul([-0.5], [0]) gradient", async () => {
    await b("mul", [[-0.5],[0]], [[0],[-0.5]], false);
});
test("mul([-0.5], [0.30000001192092896])", async () => {
    await f("mul", [[-0.5],[0.30000001192092896]], [[-0.15000000596046448]]);
});
test("mul([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("mul", [[-0.5],[0.30000001192092896]], [[0.30000001192092896],[-0.5]], false);
});
test("mul([0], [-0.5])", async () => {
    await f("mul", [[0],[-0.5]], [[0]]);
});
test("mul([0], [-0.5]) gradient", async () => {
    await b("mul", [[0],[-0.5]], [[-0.5],[0]], false);
});
test("mul([0], [0])", async () => {
    await f("mul", [[0],[0]], [[0]]);
});
test("mul([0], [0]) gradient", async () => {
    await b("mul", [[0],[0]], [[0],[0]], false);
});
test("mul([0], [0.30000001192092896])", async () => {
    await f("mul", [[0],[0.30000001192092896]], [[0]]);
});
test("mul([0], [0.30000001192092896]) gradient", async () => {
    await b("mul", [[0],[0.30000001192092896]], [[0.30000001192092896],[0]], false);
});
test("mul([0.30000001192092896], [-0.5])", async () => {
    await f("mul", [[0.30000001192092896],[-0.5]], [[-0.15000000596046448]]);
});
test("mul([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("mul", [[0.30000001192092896],[-0.5]], [[-0.5],[0.30000001192092896]], false);
});
test("mul([0.30000001192092896], [0])", async () => {
    await f("mul", [[0.30000001192092896],[0]], [[0]]);
});
test("mul([0.30000001192092896], [0]) gradient", async () => {
    await b("mul", [[0.30000001192092896],[0]], [[0],[0.30000001192092896]], false);
});
test("mul([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("mul", [[0.30000001192092896],[0.30000001192092896]], [[0.09000000357627869]]);
});
test("mul([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("mul", [[0.30000001192092896],[0.30000001192092896]], [[0.30000001192092896],[0.30000001192092896]], false);
});
test("mul_([-0.5], [-0.5])", async () => {
    await f("mul_", [[-0.5],[-0.5]], [[0.25]]);
});
test("mul_([-0.5], [0])", async () => {
    await f("mul_", [[-0.5],[0]], [[0]]);
});
test("mul_([-0.5], [0.30000001192092896])", async () => {
    await f("mul_", [[-0.5],[0.30000001192092896]], [[-0.15000000596046448]]);
});
test("mul_([0], [-0.5])", async () => {
    await f("mul_", [[0],[-0.5]], [[0]]);
});
test("mul_([0], [0])", async () => {
    await f("mul_", [[0],[0]], [[0]]);
});
test("mul_([0], [0.30000001192092896])", async () => {
    await f("mul_", [[0],[0.30000001192092896]], [[0]]);
});
test("mul_([0.30000001192092896], [-0.5])", async () => {
    await f("mul_", [[0.30000001192092896],[-0.5]], [[-0.15000000596046448]]);
});
test("mul_([0.30000001192092896], [0])", async () => {
    await f("mul_", [[0.30000001192092896],[0]], [[0]]);
});
test("mul_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("mul_", [[0.30000001192092896],[0.30000001192092896]], [[0.09000000357627869]]);
});
test("neg([-2])", async () => {
    await f("neg", [[-2]], [[2]]);
});
test("neg([-2]) gradient", async () => {
    await b("neg", [[-2]], [[-1]], false);
});
test("neg([-1])", async () => {
    await f("neg", [[-1]], [[1]]);
});
test("neg([-1]) gradient", async () => {
    await b("neg", [[-1]], [[-1]], false);
});
test("neg([-0.5])", async () => {
    await f("neg", [[-0.5]], [[0.5]]);
});
test("neg([-0.5]) gradient", async () => {
    await b("neg", [[-0.5]], [[-1]], false);
});
test("neg([0])", async () => {
    await f("neg", [[0]], [[0]]);
});
test("neg([0]) gradient", async () => {
    await b("neg", [[0]], [[-1]], false);
});
test("neg([0.5])", async () => {
    await f("neg", [[0.5]], [[-0.5]]);
});
test("neg([0.5]) gradient", async () => {
    await b("neg", [[0.5]], [[-1]], false);
});
test("neg([1])", async () => {
    await f("neg", [[1]], [[-1]]);
});
test("neg([1]) gradient", async () => {
    await b("neg", [[1]], [[-1]], false);
});
test("neg([2])", async () => {
    await f("neg", [[2]], [[-2]]);
});
test("neg([2]) gradient", async () => {
    await b("neg", [[2]], [[-1]], false);
});
test("neg_([-2])", async () => {
    await f("neg_", [[-2]], [[2]]);
});
test("neg_([-1])", async () => {
    await f("neg_", [[-1]], [[1]]);
});
test("neg_([-0.5])", async () => {
    await f("neg_", [[-0.5]], [[0.5]]);
});
test("neg_([0])", async () => {
    await f("neg_", [[0]], [[0]]);
});
test("neg_([0.5])", async () => {
    await f("neg_", [[0.5]], [[-0.5]]);
});
test("neg_([1])", async () => {
    await f("neg_", [[1]], [[-1]]);
});
test("neg_([2])", async () => {
    await f("neg_", [[2]], [[-2]]);
});
test("positive([-2])", async () => {
    await f("positive", [[-2]], [[-2]]);
});
test("positive([-2]) gradient", async () => {
    await b("positive", [[-2]], [[1]], false);
});
test("positive([-1])", async () => {
    await f("positive", [[-1]], [[-1]]);
});
test("positive([-1]) gradient", async () => {
    await b("positive", [[-1]], [[1]], false);
});
test("positive([-0.5])", async () => {
    await f("positive", [[-0.5]], [[-0.5]]);
});
test("positive([-0.5]) gradient", async () => {
    await b("positive", [[-0.5]], [[1]], false);
});
test("positive([0])", async () => {
    await f("positive", [[0]], [[0]]);
});
test("positive([0]) gradient", async () => {
    await b("positive", [[0]], [[1]], false);
});
test("positive([0.5])", async () => {
    await f("positive", [[0.5]], [[0.5]]);
});
test("positive([0.5]) gradient", async () => {
    await b("positive", [[0.5]], [[1]], false);
});
test("positive([1])", async () => {
    await f("positive", [[1]], [[1]]);
});
test("positive([1]) gradient", async () => {
    await b("positive", [[1]], [[1]], false);
});
test("positive([2])", async () => {
    await f("positive", [[2]], [[2]]);
});
test("positive([2]) gradient", async () => {
    await b("positive", [[2]], [[1]], false);
});
test("pow([-0.5], [-0.5])", async () => {
    await f("pow", [[-0.5],[-0.5]], [["NaN"]]);
});
test("pow([-0.5], [-0.5]) gradient", async () => {
    await b("pow", [[-0.5],[-0.5]], [["NaN"],["NaN"]], false);
});
test("pow([-0.5], [0])", async () => {
    await f("pow", [[-0.5],[0]], [[1]]);
});
test("pow([-0.5], [0]) gradient", async () => {
    await b("pow", [[-0.5],[0]], [[0],["NaN"]], false);
});
test("pow([-0.5], [0.30000001192092896])", async () => {
    await f("pow", [[-0.5],[0.30000001192092896]], [["NaN"]]);
});
test("pow([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("pow", [[-0.5],[0.30000001192092896]], [["NaN"],["NaN"]], false);
});
test("pow([0], [-0.5])", async () => {
    await f("pow", [[0],[-0.5]], [["+Inf"]]);
});
test("pow([0], [-0.5]) gradient", async () => {
    await b("pow", [[0],[-0.5]], [["-Inf"],["-Inf"]], false);
});
test("pow([0], [0])", async () => {
    await f("pow", [[0],[0]], [[1]]);
});
test("pow([0], [0]) gradient", async () => {
    await b("pow", [[0],[0]], [[0],[0]], false);
});
test("pow([0], [0.30000001192092896])", async () => {
    await f("pow", [[0],[0.30000001192092896]], [[0]]);
});
test("pow([0], [0.30000001192092896]) gradient", async () => {
    await b("pow", [[0],[0.30000001192092896]], [["+Inf"],[0]], false);
});
test("pow([0.30000001192092896], [-0.5])", async () => {
    await f("pow", [[0.30000001192092896],[-0.5]], [[1.8257417678833008]]);
});
test("pow([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("pow", [[0.30000001192092896],[-0.5]], [[-3.042902946472168],[-2.198143482208252]], false);
});
test("pow([0.30000001192092896], [0])", async () => {
    await f("pow", [[0.30000001192092896],[0]], [[1]]);
});
test("pow([0.30000001192092896], [0]) gradient", async () => {
    await b("pow", [[0.30000001192092896],[0]], [[0],[-1.2039728164672852]], false);
});
test("pow([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("pow", [[0.30000001192092896],[0.30000001192092896]], [[0.696845293045044]]);
});
test("pow([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("pow", [[0.30000001192092896],[0.30000001192092896]], [[0.696845293045044],[-0.8389827609062195]], false);
});
test("pow_([-0.5], [-0.5])", async () => {
    await f("pow_", [[-0.5],[-0.5]], [["NaN"]]);
});
test("pow_([-0.5], [0])", async () => {
    await f("pow_", [[-0.5],[0]], [[1]]);
});
test("pow_([-0.5], [0.30000001192092896])", async () => {
    await f("pow_", [[-0.5],[0.30000001192092896]], [["NaN"]]);
});
test("pow_([0], [-0.5])", async () => {
    await f("pow_", [[0],[-0.5]], [["+Inf"]]);
});
test("pow_([0], [0])", async () => {
    await f("pow_", [[0],[0]], [[1]]);
});
test("pow_([0], [0.30000001192092896])", async () => {
    await f("pow_", [[0],[0.30000001192092896]], [[0]]);
});
test("pow_([0.30000001192092896], [-0.5])", async () => {
    await f("pow_", [[0.30000001192092896],[-0.5]], [[1.8257417678833008]]);
});
test("pow_([0.30000001192092896], [0])", async () => {
    await f("pow_", [[0.30000001192092896],[0]], [[1]]);
});
test("pow_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("pow_", [[0.30000001192092896],[0.30000001192092896]], [[0.696845293045044]]);
});
test("rad2deg([-2])", async () => {
    await f("rad2deg", [[-2]], [[-114.59156036376953]]);
});
test("rad2deg([-2]) gradient", async () => {
    await b("rad2deg", [[-2]], [[57.295780181884766]], false);
});
test("rad2deg([-1])", async () => {
    await f("rad2deg", [[-1]], [[-57.295780181884766]]);
});
test("rad2deg([-1]) gradient", async () => {
    await b("rad2deg", [[-1]], [[57.295780181884766]], false);
});
test("rad2deg([-0.5])", async () => {
    await f("rad2deg", [[-0.5]], [[-28.647890090942383]]);
});
test("rad2deg([-0.5]) gradient", async () => {
    await b("rad2deg", [[-0.5]], [[57.295780181884766]], false);
});
test("rad2deg([0])", async () => {
    await f("rad2deg", [[0]], [[0]]);
});
test("rad2deg([0]) gradient", async () => {
    await b("rad2deg", [[0]], [[57.295780181884766]], false);
});
test("rad2deg([0.5])", async () => {
    await f("rad2deg", [[0.5]], [[28.647890090942383]]);
});
test("rad2deg([0.5]) gradient", async () => {
    await b("rad2deg", [[0.5]], [[57.295780181884766]], false);
});
test("rad2deg([1])", async () => {
    await f("rad2deg", [[1]], [[57.295780181884766]]);
});
test("rad2deg([1]) gradient", async () => {
    await b("rad2deg", [[1]], [[57.295780181884766]], false);
});
test("rad2deg([2])", async () => {
    await f("rad2deg", [[2]], [[114.59156036376953]]);
});
test("rad2deg([2]) gradient", async () => {
    await b("rad2deg", [[2]], [[57.295780181884766]], false);
});
test("rad2deg_([-2])", async () => {
    await f("rad2deg_", [[-2]], [[-114.59156036376953]]);
});
test("rad2deg_([-1])", async () => {
    await f("rad2deg_", [[-1]], [[-57.295780181884766]]);
});
test("rad2deg_([-0.5])", async () => {
    await f("rad2deg_", [[-0.5]], [[-28.647890090942383]]);
});
test("rad2deg_([0])", async () => {
    await f("rad2deg_", [[0]], [[0]]);
});
test("rad2deg_([0.5])", async () => {
    await f("rad2deg_", [[0.5]], [[28.647890090942383]]);
});
test("rad2deg_([1])", async () => {
    await f("rad2deg_", [[1]], [[57.295780181884766]]);
});
test("rad2deg_([2])", async () => {
    await f("rad2deg_", [[2]], [[114.59156036376953]]);
});
test("reciprocal([-2])", async () => {
    await f("reciprocal", [[-2]], [[-0.5]]);
});
test("reciprocal([-2]) gradient", async () => {
    await b("reciprocal", [[-2]], [[-0.25]], false);
});
test("reciprocal([-1])", async () => {
    await f("reciprocal", [[-1]], [[-1]]);
});
test("reciprocal([-1]) gradient", async () => {
    await b("reciprocal", [[-1]], [[-1]], false);
});
test("reciprocal([-0.5])", async () => {
    await f("reciprocal", [[-0.5]], [[-2]]);
});
test("reciprocal([-0.5]) gradient", async () => {
    await b("reciprocal", [[-0.5]], [[-4]], false);
});
test("reciprocal([0])", async () => {
    await f("reciprocal", [[0]], [["+Inf"]]);
});
test("reciprocal([0]) gradient", async () => {
    await b("reciprocal", [[0]], [["-Inf"]], false);
});
test("reciprocal([0.5])", async () => {
    await f("reciprocal", [[0.5]], [[2]]);
});
test("reciprocal([0.5]) gradient", async () => {
    await b("reciprocal", [[0.5]], [[-4]], false);
});
test("reciprocal([1])", async () => {
    await f("reciprocal", [[1]], [[1]]);
});
test("reciprocal([1]) gradient", async () => {
    await b("reciprocal", [[1]], [[-1]], false);
});
test("reciprocal([2])", async () => {
    await f("reciprocal", [[2]], [[0.5]]);
});
test("reciprocal([2]) gradient", async () => {
    await b("reciprocal", [[2]], [[-0.25]], false);
});
test("reciprocal_([-2])", async () => {
    await f("reciprocal_", [[-2]], [[-0.5]]);
});
test("reciprocal_([-1])", async () => {
    await f("reciprocal_", [[-1]], [[-1]]);
});
test("reciprocal_([-0.5])", async () => {
    await f("reciprocal_", [[-0.5]], [[-2]]);
});
test("reciprocal_([0])", async () => {
    await f("reciprocal_", [[0]], [["+Inf"]]);
});
test("reciprocal_([0.5])", async () => {
    await f("reciprocal_", [[0.5]], [[2]]);
});
test("reciprocal_([1])", async () => {
    await f("reciprocal_", [[1]], [[1]]);
});
test("reciprocal_([2])", async () => {
    await f("reciprocal_", [[2]], [[0.5]]);
});
test("relu([-2])", async () => {
    await f("relu", [[-2]], [[0]]);
});
test("relu([-2]) gradient", async () => {
    await b("relu", [[-2]], [[0]], false);
});
test("relu([-1])", async () => {
    await f("relu", [[-1]], [[0]]);
});
test("relu([-1]) gradient", async () => {
    await b("relu", [[-1]], [[0]], false);
});
test("relu([-0.5])", async () => {
    await f("relu", [[-0.5]], [[0]]);
});
test("relu([-0.5]) gradient", async () => {
    await b("relu", [[-0.5]], [[0]], false);
});
test("relu([0])", async () => {
    await f("relu", [[0]], [[0]]);
});
test("relu([0]) gradient", async () => {
    await b("relu", [[0]], [[0]], false);
});
test("relu([0.5])", async () => {
    await f("relu", [[0.5]], [[0.5]]);
});
test("relu([0.5]) gradient", async () => {
    await b("relu", [[0.5]], [[1]], false);
});
test("relu([1])", async () => {
    await f("relu", [[1]], [[1]]);
});
test("relu([1]) gradient", async () => {
    await b("relu", [[1]], [[1]], false);
});
test("relu([2])", async () => {
    await f("relu", [[2]], [[2]]);
});
test("relu([2]) gradient", async () => {
    await b("relu", [[2]], [[1]], false);
});
test("relu_([-2])", async () => {
    await f("relu_", [[-2]], [[0]]);
});
test("relu_([-1])", async () => {
    await f("relu_", [[-1]], [[0]]);
});
test("relu_([-0.5])", async () => {
    await f("relu_", [[-0.5]], [[0]]);
});
test("relu_([0])", async () => {
    await f("relu_", [[0]], [[0]]);
});
test("relu_([0.5])", async () => {
    await f("relu_", [[0.5]], [[0.5]]);
});
test("relu_([1])", async () => {
    await f("relu_", [[1]], [[1]]);
});
test("relu_([2])", async () => {
    await f("relu_", [[2]], [[2]]);
});
test("round([-2])", async () => {
    await f("round", [[-2]], [[-2]]);
});
test("round([-2]) gradient", async () => {
    await b("round", [[-2]], [[0]], false);
});
test("round([-1])", async () => {
    await f("round", [[-1]], [[-1]]);
});
test("round([-1]) gradient", async () => {
    await b("round", [[-1]], [[0]], false);
});
test("round([-0.5])", async () => {
    await f("round", [[-0.5]], [[0]]);
});
test("round([-0.5]) gradient", async () => {
    await b("round", [[-0.5]], [[0]], false);
});
test("round([0])", async () => {
    await f("round", [[0]], [[0]]);
});
test("round([0]) gradient", async () => {
    await b("round", [[0]], [[0]], false);
});
test("round([0.5])", async () => {
    await f("round", [[0.5]], [[0]]);
});
test("round([0.5]) gradient", async () => {
    await b("round", [[0.5]], [[0]], false);
});
test("round([1])", async () => {
    await f("round", [[1]], [[1]]);
});
test("round([1]) gradient", async () => {
    await b("round", [[1]], [[0]], false);
});
test("round([2])", async () => {
    await f("round", [[2]], [[2]]);
});
test("round([2]) gradient", async () => {
    await b("round", [[2]], [[0]], false);
});
test("round_([-2])", async () => {
    await f("round_", [[-2]], [[-2]]);
});
test("round_([-1])", async () => {
    await f("round_", [[-1]], [[-1]]);
});
test("round_([-0.5])", async () => {
    await f("round_", [[-0.5]], [[0]]);
});
test("round_([0])", async () => {
    await f("round_", [[0]], [[0]]);
});
test("round_([0.5])", async () => {
    await f("round_", [[0.5]], [[0]]);
});
test("round_([1])", async () => {
    await f("round_", [[1]], [[1]]);
});
test("round_([2])", async () => {
    await f("round_", [[2]], [[2]]);
});
test("rsqrt([-2])", async () => {
    await f("rsqrt", [[-2]], [["NaN"]]);
});
test("rsqrt([-2]) gradient", async () => {
    await b("rsqrt", [[-2]], [["NaN"]], false);
});
test("rsqrt([-1])", async () => {
    await f("rsqrt", [[-1]], [["NaN"]]);
});
test("rsqrt([-1]) gradient", async () => {
    await b("rsqrt", [[-1]], [["NaN"]], false);
});
test("rsqrt([-0.5])", async () => {
    await f("rsqrt", [[-0.5]], [["NaN"]]);
});
test("rsqrt([-0.5]) gradient", async () => {
    await b("rsqrt", [[-0.5]], [["NaN"]], false);
});
test("rsqrt([0])", async () => {
    await f("rsqrt", [[0]], [["+Inf"]]);
});
test("rsqrt([0]) gradient", async () => {
    await b("rsqrt", [[0]], [["-Inf"]], false);
});
test("rsqrt([0.5])", async () => {
    await f("rsqrt", [[0.5]], [[1.4142135381698608]]);
});
test("rsqrt([0.5]) gradient", async () => {
    await b("rsqrt", [[0.5]], [[-1.4142134189605713]], false);
});
test("rsqrt([1])", async () => {
    await f("rsqrt", [[1]], [[1]]);
});
test("rsqrt([1]) gradient", async () => {
    await b("rsqrt", [[1]], [[-0.5]], false);
});
test("rsqrt([2])", async () => {
    await f("rsqrt", [[2]], [[0.7071067690849304]]);
});
test("rsqrt([2]) gradient", async () => {
    await b("rsqrt", [[2]], [[-0.1767766773700714]], false);
});
test("rsqrt_([-2])", async () => {
    await f("rsqrt_", [[-2]], [["NaN"]]);
});
test("rsqrt_([-1])", async () => {
    await f("rsqrt_", [[-1]], [["NaN"]]);
});
test("rsqrt_([-0.5])", async () => {
    await f("rsqrt_", [[-0.5]], [["NaN"]]);
});
test("rsqrt_([0])", async () => {
    await f("rsqrt_", [[0]], [["+Inf"]]);
});
test("rsqrt_([0.5])", async () => {
    await f("rsqrt_", [[0.5]], [[1.4142135381698608]]);
});
test("rsqrt_([1])", async () => {
    await f("rsqrt_", [[1]], [[1]]);
});
test("rsqrt_([2])", async () => {
    await f("rsqrt_", [[2]], [[0.7071067690849304]]);
});
test("sigmoid([-2])", async () => {
    await f("sigmoid", [[-2]], [[0.11920291930437088]]);
});
test("sigmoid([-2]) gradient", async () => {
    await b("sigmoid", [[-2]], [[0.10499358177185059]], false);
});
test("sigmoid([-1])", async () => {
    await f("sigmoid", [[-1]], [[0.2689414322376251]]);
});
test("sigmoid([-1]) gradient", async () => {
    await b("sigmoid", [[-1]], [[0.1966119408607483]], false);
});
test("sigmoid([-0.5])", async () => {
    await f("sigmoid", [[-0.5]], [[0.3775406777858734]]);
});
test("sigmoid([-0.5]) gradient", async () => {
    await b("sigmoid", [[-0.5]], [[0.23500370979309082]], false);
});
test("sigmoid([0])", async () => {
    await f("sigmoid", [[0]], [[0.5]]);
});
test("sigmoid([0]) gradient", async () => {
    await b("sigmoid", [[0]], [[0.25]], false);
});
test("sigmoid([0.5])", async () => {
    await f("sigmoid", [[0.5]], [[0.622459352016449]]);
});
test("sigmoid([0.5]) gradient", async () => {
    await b("sigmoid", [[0.5]], [[0.23500370979309082]], false);
});
test("sigmoid([1])", async () => {
    await f("sigmoid", [[1]], [[0.7310585975646973]]);
});
test("sigmoid([1]) gradient", async () => {
    await b("sigmoid", [[1]], [[0.1966119259595871]], false);
});
test("sigmoid([2])", async () => {
    await f("sigmoid", [[2]], [[0.8807970285415649]]);
});
test("sigmoid([2]) gradient", async () => {
    await b("sigmoid", [[2]], [[0.10499362647533417]], false);
});
test("sigmoid_([-2])", async () => {
    await f("sigmoid_", [[-2]], [[0.11920291930437088]]);
});
test("sigmoid_([-1])", async () => {
    await f("sigmoid_", [[-1]], [[0.2689414322376251]]);
});
test("sigmoid_([-0.5])", async () => {
    await f("sigmoid_", [[-0.5]], [[0.3775406777858734]]);
});
test("sigmoid_([0])", async () => {
    await f("sigmoid_", [[0]], [[0.5]]);
});
test("sigmoid_([0.5])", async () => {
    await f("sigmoid_", [[0.5]], [[0.622459352016449]]);
});
test("sigmoid_([1])", async () => {
    await f("sigmoid_", [[1]], [[0.7310585975646973]]);
});
test("sigmoid_([2])", async () => {
    await f("sigmoid_", [[2]], [[0.8807970285415649]]);
});
test("sign([-2])", async () => {
    await f("sign", [[-2]], [[-1]]);
});
test("sign([-2]) gradient", async () => {
    await b("sign", [[-2]], [[0]], false);
});
test("sign([-1])", async () => {
    await f("sign", [[-1]], [[-1]]);
});
test("sign([-1]) gradient", async () => {
    await b("sign", [[-1]], [[0]], false);
});
test("sign([-0.5])", async () => {
    await f("sign", [[-0.5]], [[-1]]);
});
test("sign([-0.5]) gradient", async () => {
    await b("sign", [[-0.5]], [[0]], false);
});
test("sign([0])", async () => {
    await f("sign", [[0]], [[0]]);
});
test("sign([0]) gradient", async () => {
    await b("sign", [[0]], [[0]], false);
});
test("sign([0.5])", async () => {
    await f("sign", [[0.5]], [[1]]);
});
test("sign([0.5]) gradient", async () => {
    await b("sign", [[0.5]], [[0]], false);
});
test("sign([1])", async () => {
    await f("sign", [[1]], [[1]]);
});
test("sign([1]) gradient", async () => {
    await b("sign", [[1]], [[0]], false);
});
test("sign([2])", async () => {
    await f("sign", [[2]], [[1]]);
});
test("sign([2]) gradient", async () => {
    await b("sign", [[2]], [[0]], false);
});
test("sign_([-2])", async () => {
    await f("sign_", [[-2]], [[-1]]);
});
test("sign_([-1])", async () => {
    await f("sign_", [[-1]], [[-1]]);
});
test("sign_([-0.5])", async () => {
    await f("sign_", [[-0.5]], [[-1]]);
});
test("sign_([0])", async () => {
    await f("sign_", [[0]], [[0]]);
});
test("sign_([0.5])", async () => {
    await f("sign_", [[0.5]], [[1]]);
});
test("sign_([1])", async () => {
    await f("sign_", [[1]], [[1]]);
});
test("sign_([2])", async () => {
    await f("sign_", [[2]], [[1]]);
});
test("silu([-2])", async () => {
    await f("silu", [[-2]], [[-0.23840583860874176]]);
});
test("silu([-2]) gradient", async () => {
    await b("silu", [[-2]], [[-0.09078425168991089]], false);
});
test("silu([-1])", async () => {
    await f("silu", [[-1]], [[-0.2689414322376251]]);
});
test("silu([-1]) gradient", async () => {
    await b("silu", [[-1]], [[0.07232948392629623]], false);
});
test("silu([-0.5])", async () => {
    await f("silu", [[-0.5]], [[-0.1887703388929367]]);
});
test("silu([-0.5]) gradient", async () => {
    await b("silu", [[-0.5]], [[0.260038822889328]], false);
});
test("silu([0])", async () => {
    await f("silu", [[0]], [[0]]);
});
test("silu([0]) gradient", async () => {
    await b("silu", [[0]], [[0.5]], false);
});
test("silu([0.5])", async () => {
    await f("silu", [[0.5]], [[0.3112296760082245]]);
});
test("silu([0.5]) gradient", async () => {
    await b("silu", [[0.5]], [[0.7399612069129944]], false);
});
test("silu([1])", async () => {
    await f("silu", [[1]], [[0.7310585975646973]]);
});
test("silu([1]) gradient", async () => {
    await b("silu", [[1]], [[0.9276705384254456]], false);
});
test("silu([2])", async () => {
    await f("silu", [[2]], [[1.7615940570831299]]);
});
test("silu([2]) gradient", async () => {
    await b("silu", [[2]], [[1.0907843112945557]], false);
});
test("silu_([-2])", async () => {
    await f("silu_", [[-2]], [[-0.23840583860874176]]);
});
test("silu_([-1])", async () => {
    await f("silu_", [[-1]], [[-0.2689414322376251]]);
});
test("silu_([-0.5])", async () => {
    await f("silu_", [[-0.5]], [[-0.1887703388929367]]);
});
test("silu_([0])", async () => {
    await f("silu_", [[0]], [[0]]);
});
test("silu_([0.5])", async () => {
    await f("silu_", [[0.5]], [[0.3112296760082245]]);
});
test("silu_([1])", async () => {
    await f("silu_", [[1]], [[0.7310585975646973]]);
});
test("silu_([2])", async () => {
    await f("silu_", [[2]], [[1.7615940570831299]]);
});
test("sin([-2])", async () => {
    await f("sin", [[-2]], [[-0.9092974066734314]]);
});
test("sin([-2]) gradient", async () => {
    await b("sin", [[-2]], [[-0.416146844625473]], false);
});
test("sin([-1])", async () => {
    await f("sin", [[-1]], [[-0.8414709568023682]]);
});
test("sin([-1]) gradient", async () => {
    await b("sin", [[-1]], [[0.5403023362159729]], false);
});
test("sin([-0.5])", async () => {
    await f("sin", [[-0.5]], [[-0.4794255495071411]]);
});
test("sin([-0.5]) gradient", async () => {
    await b("sin", [[-0.5]], [[0.8775825500488281]], false);
});
test("sin([0])", async () => {
    await f("sin", [[0]], [[0]]);
});
test("sin([0]) gradient", async () => {
    await b("sin", [[0]], [[1]], false);
});
test("sin([0.5])", async () => {
    await f("sin", [[0.5]], [[0.4794255495071411]]);
});
test("sin([0.5]) gradient", async () => {
    await b("sin", [[0.5]], [[0.8775825500488281]], false);
});
test("sin([1])", async () => {
    await f("sin", [[1]], [[0.8414709568023682]]);
});
test("sin([1]) gradient", async () => {
    await b("sin", [[1]], [[0.5403023362159729]], false);
});
test("sin([2])", async () => {
    await f("sin", [[2]], [[0.9092974066734314]]);
});
test("sin([2]) gradient", async () => {
    await b("sin", [[2]], [[-0.416146844625473]], false);
});
test("sin_([-2])", async () => {
    await f("sin_", [[-2]], [[-0.9092974066734314]]);
});
test("sin_([-1])", async () => {
    await f("sin_", [[-1]], [[-0.8414709568023682]]);
});
test("sin_([-0.5])", async () => {
    await f("sin_", [[-0.5]], [[-0.4794255495071411]]);
});
test("sin_([0])", async () => {
    await f("sin_", [[0]], [[0]]);
});
test("sin_([0.5])", async () => {
    await f("sin_", [[0.5]], [[0.4794255495071411]]);
});
test("sin_([1])", async () => {
    await f("sin_", [[1]], [[0.8414709568023682]]);
});
test("sin_([2])", async () => {
    await f("sin_", [[2]], [[0.9092974066734314]]);
});
test("sinc([-2])", async () => {
    await f("sinc", [[-2]], [[2.7827534054836178e-8]]);
});
test("sinc([-2]) gradient", async () => {
    await b("sinc", [[-2]], [[-0.5]], false);
});
test("sinc([-1])", async () => {
    await f("sinc", [[-1]], [[-2.7827534054836178e-8]]);
});
test("sinc([-1]) gradient", async () => {
    await b("sinc", [[-1]], [[1]], false);
});
test("sinc([-0.5])", async () => {
    await f("sinc", [[-0.5]], [[0.6366197466850281]]);
});
test("sinc([-0.5]) gradient", async () => {
    await b("sinc", [[-0.5]], [[1.2732396125793457]], false);
});
test("sinc([0])", async () => {
    await f("sinc", [[0]], [[1]]);
});
test("sinc([0]) gradient", async () => {
    await b("sinc", [[0]], [[0]], false);
});
test("sinc([0.5])", async () => {
    await f("sinc", [[0.5]], [[0.6366197466850281]]);
});
test("sinc([0.5]) gradient", async () => {
    await b("sinc", [[0.5]], [[-1.2732396125793457]], false);
});
test("sinc([1])", async () => {
    await f("sinc", [[1]], [[-2.7827534054836178e-8]]);
});
test("sinc([1]) gradient", async () => {
    await b("sinc", [[1]], [[-1]], false);
});
test("sinc([2])", async () => {
    await f("sinc", [[2]], [[2.7827534054836178e-8]]);
});
test("sinc([2]) gradient", async () => {
    await b("sinc", [[2]], [[0.5]], false);
});
test("sinc_([-2])", async () => {
    await f("sinc_", [[-2]], [[2.7827534054836178e-8]]);
});
test("sinc_([-1])", async () => {
    await f("sinc_", [[-1]], [[-2.7827534054836178e-8]]);
});
test("sinc_([-0.5])", async () => {
    await f("sinc_", [[-0.5]], [[0.6366197466850281]]);
});
test("sinc_([0])", async () => {
    await f("sinc_", [[0]], [[1]]);
});
test("sinc_([0.5])", async () => {
    await f("sinc_", [[0.5]], [[0.6366197466850281]]);
});
test("sinc_([1])", async () => {
    await f("sinc_", [[1]], [[-2.7827534054836178e-8]]);
});
test("sinc_([2])", async () => {
    await f("sinc_", [[2]], [[2.7827534054836178e-8]]);
});
test("sinh([-2])", async () => {
    await f("sinh", [[-2]], [[-3.6268603801727295]]);
});
test("sinh([-2]) gradient", async () => {
    await b("sinh", [[-2]], [[3.762195587158203]], false);
});
test("sinh([-1])", async () => {
    await f("sinh", [[-1]], [[-1.175201177597046]]);
});
test("sinh([-1]) gradient", async () => {
    await b("sinh", [[-1]], [[1.5430806875228882]], false);
});
test("sinh([-0.5])", async () => {
    await f("sinh", [[-0.5]], [[-0.5210952758789062]]);
});
test("sinh([-0.5]) gradient", async () => {
    await b("sinh", [[-0.5]], [[1.1276259422302246]], false);
});
test("sinh([0])", async () => {
    await f("sinh", [[0]], [[0]]);
});
test("sinh([0]) gradient", async () => {
    await b("sinh", [[0]], [[1]], false);
});
test("sinh([0.5])", async () => {
    await f("sinh", [[0.5]], [[0.5210952758789062]]);
});
test("sinh([0.5]) gradient", async () => {
    await b("sinh", [[0.5]], [[1.1276259422302246]], false);
});
test("sinh([1])", async () => {
    await f("sinh", [[1]], [[1.175201177597046]]);
});
test("sinh([1]) gradient", async () => {
    await b("sinh", [[1]], [[1.5430806875228882]], false);
});
test("sinh([2])", async () => {
    await f("sinh", [[2]], [[3.6268603801727295]]);
});
test("sinh([2]) gradient", async () => {
    await b("sinh", [[2]], [[3.762195587158203]], false);
});
test("sinh_([-2])", async () => {
    await f("sinh_", [[-2]], [[-3.6268603801727295]]);
});
test("sinh_([-1])", async () => {
    await f("sinh_", [[-1]], [[-1.175201177597046]]);
});
test("sinh_([-0.5])", async () => {
    await f("sinh_", [[-0.5]], [[-0.5210952758789062]]);
});
test("sinh_([0])", async () => {
    await f("sinh_", [[0]], [[0]]);
});
test("sinh_([0.5])", async () => {
    await f("sinh_", [[0.5]], [[0.5210952758789062]]);
});
test("sinh_([1])", async () => {
    await f("sinh_", [[1]], [[1.175201177597046]]);
});
test("sinh_([2])", async () => {
    await f("sinh_", [[2]], [[3.6268603801727295]]);
});
test("sqrt([-2])", async () => {
    await f("sqrt", [[-2]], [["NaN"]]);
});
test("sqrt([-2]) gradient", async () => {
    await b("sqrt", [[-2]], [["NaN"]], false);
});
test("sqrt([-1])", async () => {
    await f("sqrt", [[-1]], [["NaN"]]);
});
test("sqrt([-1]) gradient", async () => {
    await b("sqrt", [[-1]], [["NaN"]], false);
});
test("sqrt([-0.5])", async () => {
    await f("sqrt", [[-0.5]], [["NaN"]]);
});
test("sqrt([-0.5]) gradient", async () => {
    await b("sqrt", [[-0.5]], [["NaN"]], false);
});
test("sqrt([0])", async () => {
    await f("sqrt", [[0]], [[0]]);
});
test("sqrt([0]) gradient", async () => {
    await b("sqrt", [[0]], [["+Inf"]], false);
});
test("sqrt([0.5])", async () => {
    await f("sqrt", [[0.5]], [[0.7071067690849304]]);
});
test("sqrt([0.5]) gradient", async () => {
    await b("sqrt", [[0.5]], [[0.7071067690849304]], false);
});
test("sqrt([1])", async () => {
    await f("sqrt", [[1]], [[1]]);
});
test("sqrt([1]) gradient", async () => {
    await b("sqrt", [[1]], [[0.5]], false);
});
test("sqrt([2])", async () => {
    await f("sqrt", [[2]], [[1.4142135381698608]]);
});
test("sqrt([2]) gradient", async () => {
    await b("sqrt", [[2]], [[0.3535533845424652]], false);
});
test("sqrt_([-2])", async () => {
    await f("sqrt_", [[-2]], [["NaN"]]);
});
test("sqrt_([-1])", async () => {
    await f("sqrt_", [[-1]], [["NaN"]]);
});
test("sqrt_([-0.5])", async () => {
    await f("sqrt_", [[-0.5]], [["NaN"]]);
});
test("sqrt_([0])", async () => {
    await f("sqrt_", [[0]], [[0]]);
});
test("sqrt_([0.5])", async () => {
    await f("sqrt_", [[0.5]], [[0.7071067690849304]]);
});
test("sqrt_([1])", async () => {
    await f("sqrt_", [[1]], [[1]]);
});
test("sqrt_([2])", async () => {
    await f("sqrt_", [[2]], [[1.4142135381698608]]);
});
test("square([-2])", async () => {
    await f("square", [[-2]], [[4]]);
});
test("square([-2]) gradient", async () => {
    await b("square", [[-2]], [[-4]], false);
});
test("square([-1])", async () => {
    await f("square", [[-1]], [[1]]);
});
test("square([-1]) gradient", async () => {
    await b("square", [[-1]], [[-2]], false);
});
test("square([-0.5])", async () => {
    await f("square", [[-0.5]], [[0.25]]);
});
test("square([-0.5]) gradient", async () => {
    await b("square", [[-0.5]], [[-1]], false);
});
test("square([0])", async () => {
    await f("square", [[0]], [[0]]);
});
test("square([0]) gradient", async () => {
    await b("square", [[0]], [[0]], false);
});
test("square([0.5])", async () => {
    await f("square", [[0.5]], [[0.25]]);
});
test("square([0.5]) gradient", async () => {
    await b("square", [[0.5]], [[1]], false);
});
test("square([1])", async () => {
    await f("square", [[1]], [[1]]);
});
test("square([1]) gradient", async () => {
    await b("square", [[1]], [[2]], false);
});
test("square([2])", async () => {
    await f("square", [[2]], [[4]]);
});
test("square([2]) gradient", async () => {
    await b("square", [[2]], [[4]], false);
});
test("square_([-2])", async () => {
    await f("square_", [[-2]], [[4]]);
});
test("square_([-1])", async () => {
    await f("square_", [[-1]], [[1]]);
});
test("square_([-0.5])", async () => {
    await f("square_", [[-0.5]], [[0.25]]);
});
test("square_([0])", async () => {
    await f("square_", [[0]], [[0]]);
});
test("square_([0.5])", async () => {
    await f("square_", [[0.5]], [[0.25]]);
});
test("square_([1])", async () => {
    await f("square_", [[1]], [[1]]);
});
test("square_([2])", async () => {
    await f("square_", [[2]], [[4]]);
});
test("sub([-0.5], [-0.5])", async () => {
    await f("sub", [[-0.5],[-0.5]], [[0]]);
});
test("sub([-0.5], [-0.5]) gradient", async () => {
    await b("sub", [[-0.5],[-0.5]], [[1],[-1]], false);
});
test("sub([-0.5], [0])", async () => {
    await f("sub", [[-0.5],[0]], [[-0.5]]);
});
test("sub([-0.5], [0]) gradient", async () => {
    await b("sub", [[-0.5],[0]], [[1],[-1]], false);
});
test("sub([-0.5], [0.30000001192092896])", async () => {
    await f("sub", [[-0.5],[0.30000001192092896]], [[-0.800000011920929]]);
});
test("sub([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("sub", [[-0.5],[0.30000001192092896]], [[1],[-1]], false);
});
test("sub([0], [-0.5])", async () => {
    await f("sub", [[0],[-0.5]], [[0.5]]);
});
test("sub([0], [-0.5]) gradient", async () => {
    await b("sub", [[0],[-0.5]], [[1],[-1]], false);
});
test("sub([0], [0])", async () => {
    await f("sub", [[0],[0]], [[0]]);
});
test("sub([0], [0]) gradient", async () => {
    await b("sub", [[0],[0]], [[1],[-1]], false);
});
test("sub([0], [0.30000001192092896])", async () => {
    await f("sub", [[0],[0.30000001192092896]], [[-0.30000001192092896]]);
});
test("sub([0], [0.30000001192092896]) gradient", async () => {
    await b("sub", [[0],[0.30000001192092896]], [[1],[-1]], false);
});
test("sub([0.30000001192092896], [-0.5])", async () => {
    await f("sub", [[0.30000001192092896],[-0.5]], [[0.800000011920929]]);
});
test("sub([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("sub", [[0.30000001192092896],[-0.5]], [[1],[-1]], false);
});
test("sub([0.30000001192092896], [0])", async () => {
    await f("sub", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("sub([0.30000001192092896], [0]) gradient", async () => {
    await b("sub", [[0.30000001192092896],[0]], [[1],[-1]], false);
});
test("sub([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("sub", [[0.30000001192092896],[0.30000001192092896]], [[0]]);
});
test("sub([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("sub", [[0.30000001192092896],[0.30000001192092896]], [[1],[-1]], false);
});
test("sub_([-0.5], [-0.5])", async () => {
    await f("sub_", [[-0.5],[-0.5]], [[0]]);
});
test("sub_([-0.5], [0])", async () => {
    await f("sub_", [[-0.5],[0]], [[-0.5]]);
});
test("sub_([-0.5], [0.30000001192092896])", async () => {
    await f("sub_", [[-0.5],[0.30000001192092896]], [[-0.800000011920929]]);
});
test("sub_([0], [-0.5])", async () => {
    await f("sub_", [[0],[-0.5]], [[0.5]]);
});
test("sub_([0], [0])", async () => {
    await f("sub_", [[0],[0]], [[0]]);
});
test("sub_([0], [0.30000001192092896])", async () => {
    await f("sub_", [[0],[0.30000001192092896]], [[-0.30000001192092896]]);
});
test("sub_([0.30000001192092896], [-0.5])", async () => {
    await f("sub_", [[0.30000001192092896],[-0.5]], [[0.800000011920929]]);
});
test("sub_([0.30000001192092896], [0])", async () => {
    await f("sub_", [[0.30000001192092896],[0]], [[0.30000001192092896]]);
});
test("sub_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("sub_", [[0.30000001192092896],[0.30000001192092896]], [[0]]);
});
test("tan([-2])", async () => {
    await f("tan", [[-2]], [[2.185039758682251]]);
});
test("tan([-2]) gradient", async () => {
    await b("tan", [[-2]], [[5.7743988037109375]], false);
});
test("tan([-1])", async () => {
    await f("tan", [[-1]], [[-1.5574077367782593]]);
});
test("tan([-1]) gradient", async () => {
    await b("tan", [[-1]], [[3.425518751144409]], false);
});
test("tan([-0.5])", async () => {
    await f("tan", [[-0.5]], [[-0.5463024973869324]]);
});
test("tan([-0.5]) gradient", async () => {
    await b("tan", [[-0.5]], [[1.2984464168548584]], false);
});
test("tan([0])", async () => {
    await f("tan", [[0]], [[0]]);
});
test("tan([0]) gradient", async () => {
    await b("tan", [[0]], [[1]], false);
});
test("tan([0.5])", async () => {
    await f("tan", [[0.5]], [[0.5463024973869324]]);
});
test("tan([0.5]) gradient", async () => {
    await b("tan", [[0.5]], [[1.2984464168548584]], false);
});
test("tan([1])", async () => {
    await f("tan", [[1]], [[1.5574077367782593]]);
});
test("tan([1]) gradient", async () => {
    await b("tan", [[1]], [[3.425518751144409]], false);
});
test("tan([2])", async () => {
    await f("tan", [[2]], [[-2.185039758682251]]);
});
test("tan([2]) gradient", async () => {
    await b("tan", [[2]], [[5.7743988037109375]], false);
});
test("tan_([-2])", async () => {
    await f("tan_", [[-2]], [[2.185039758682251]]);
});
test("tan_([-1])", async () => {
    await f("tan_", [[-1]], [[-1.5574077367782593]]);
});
test("tan_([-0.5])", async () => {
    await f("tan_", [[-0.5]], [[-0.5463024973869324]]);
});
test("tan_([0])", async () => {
    await f("tan_", [[0]], [[0]]);
});
test("tan_([0.5])", async () => {
    await f("tan_", [[0.5]], [[0.5463024973869324]]);
});
test("tan_([1])", async () => {
    await f("tan_", [[1]], [[1.5574077367782593]]);
});
test("tan_([2])", async () => {
    await f("tan_", [[2]], [[-2.185039758682251]]);
});
test("tanh([-2])", async () => {
    await f("tanh", [[-2]], [[-0.9640275835990906]]);
});
test("tanh([-2]) gradient", async () => {
    await b("tanh", [[-2]], [[0.07065081596374512]], false);
});
test("tanh([-1])", async () => {
    await f("tanh", [[-1]], [[-0.7615941762924194]]);
});
test("tanh([-1]) gradient", async () => {
    await b("tanh", [[-1]], [[0.41997432708740234]], false);
});
test("tanh([-0.5])", async () => {
    await f("tanh", [[-0.5]], [[-0.46211716532707214]]);
});
test("tanh([-0.5]) gradient", async () => {
    await b("tanh", [[-0.5]], [[0.7864477038383484]], false);
});
test("tanh([0])", async () => {
    await f("tanh", [[0]], [[0]]);
});
test("tanh([0]) gradient", async () => {
    await b("tanh", [[0]], [[1]], false);
});
test("tanh([0.5])", async () => {
    await f("tanh", [[0.5]], [[0.46211716532707214]]);
});
test("tanh([0.5]) gradient", async () => {
    await b("tanh", [[0.5]], [[0.7864477038383484]], false);
});
test("tanh([1])", async () => {
    await f("tanh", [[1]], [[0.7615941762924194]]);
});
test("tanh([1]) gradient", async () => {
    await b("tanh", [[1]], [[0.41997432708740234]], false);
});
test("tanh([2])", async () => {
    await f("tanh", [[2]], [[0.9640275835990906]]);
});
test("tanh([2]) gradient", async () => {
    await b("tanh", [[2]], [[0.07065081596374512]], false);
});
test("tanh_([-2])", async () => {
    await f("tanh_", [[-2]], [[-0.9640275835990906]]);
});
test("tanh_([-1])", async () => {
    await f("tanh_", [[-1]], [[-0.7615941762924194]]);
});
test("tanh_([-0.5])", async () => {
    await f("tanh_", [[-0.5]], [[-0.46211716532707214]]);
});
test("tanh_([0])", async () => {
    await f("tanh_", [[0]], [[0]]);
});
test("tanh_([0.5])", async () => {
    await f("tanh_", [[0.5]], [[0.46211716532707214]]);
});
test("tanh_([1])", async () => {
    await f("tanh_", [[1]], [[0.7615941762924194]]);
});
test("tanh_([2])", async () => {
    await f("tanh_", [[2]], [[0.9640275835990906]]);
});
test("trunc([-2])", async () => {
    await f("trunc", [[-2]], [[-2]]);
});
test("trunc([-2]) gradient", async () => {
    await b("trunc", [[-2]], [[0]], false);
});
test("trunc([-1])", async () => {
    await f("trunc", [[-1]], [[-1]]);
});
test("trunc([-1]) gradient", async () => {
    await b("trunc", [[-1]], [[0]], false);
});
test("trunc([-0.5])", async () => {
    await f("trunc", [[-0.5]], [[0]]);
});
test("trunc([-0.5]) gradient", async () => {
    await b("trunc", [[-0.5]], [[0]], false);
});
test("trunc([0])", async () => {
    await f("trunc", [[0]], [[0]]);
});
test("trunc([0]) gradient", async () => {
    await b("trunc", [[0]], [[0]], false);
});
test("trunc([0.5])", async () => {
    await f("trunc", [[0.5]], [[0]]);
});
test("trunc([0.5]) gradient", async () => {
    await b("trunc", [[0.5]], [[0]], false);
});
test("trunc([1])", async () => {
    await f("trunc", [[1]], [[1]]);
});
test("trunc([1]) gradient", async () => {
    await b("trunc", [[1]], [[0]], false);
});
test("trunc([2])", async () => {
    await f("trunc", [[2]], [[2]]);
});
test("trunc([2]) gradient", async () => {
    await b("trunc", [[2]], [[0]], false);
});
test("trunc_([-2])", async () => {
    await f("trunc_", [[-2]], [[-2]]);
});
test("trunc_([-1])", async () => {
    await f("trunc_", [[-1]], [[-1]]);
});
test("trunc_([-0.5])", async () => {
    await f("trunc_", [[-0.5]], [[0]]);
});
test("trunc_([0])", async () => {
    await f("trunc_", [[0]], [[0]]);
});
test("trunc_([0.5])", async () => {
    await f("trunc_", [[0.5]], [[0]]);
});
test("trunc_([1])", async () => {
    await f("trunc_", [[1]], [[1]]);
});
test("trunc_([2])", async () => {
    await f("trunc_", [[2]], [[2]]);
});
test("xlogy([-0.5], [-0.5])", async () => {
    await f("xlogy", [[-0.5],[-0.5]], [["NaN"]]);
});
test("xlogy([-0.5], [-0.5]) gradient", async () => {
    await b("xlogy", [[-0.5],[-0.5]], [["NaN"],[1]], false);
});
test("xlogy([-0.5], [0])", async () => {
    await f("xlogy", [[-0.5],[0]], [["+Inf"]]);
});
test("xlogy([-0.5], [0]) gradient", async () => {
    await b("xlogy", [[-0.5],[0]], [["-Inf"],["-Inf"]], false);
});
test("xlogy([-0.5], [0.30000001192092896])", async () => {
    await f("xlogy", [[-0.5],[0.30000001192092896]], [[0.6019864082336426]]);
});
test("xlogy([-0.5], [0.30000001192092896]) gradient", async () => {
    await b("xlogy", [[-0.5],[0.30000001192092896]], [[-1.2039728164672852],[-1.6666666269302368]], false);
});
test("xlogy([0], [-0.5])", async () => {
    await f("xlogy", [[0],[-0.5]], [[0]]);
});
test("xlogy([0], [-0.5]) gradient", async () => {
    await b("xlogy", [[0],[-0.5]], [[0],[0]], false);
});
test("xlogy([0], [0])", async () => {
    await f("xlogy", [[0],[0]], [[0]]);
});
test("xlogy([0], [0]) gradient", async () => {
    await b("xlogy", [[0],[0]], [[0],[0]], false);
});
test("xlogy([0], [0.30000001192092896])", async () => {
    await f("xlogy", [[0],[0.30000001192092896]], [[0]]);
});
test("xlogy([0], [0.30000001192092896]) gradient", async () => {
    await b("xlogy", [[0],[0.30000001192092896]], [[0],[0]], false);
});
test("xlogy([0.30000001192092896], [-0.5])", async () => {
    await f("xlogy", [[0.30000001192092896],[-0.5]], [["NaN"]]);
});
test("xlogy([0.30000001192092896], [-0.5]) gradient", async () => {
    await b("xlogy", [[0.30000001192092896],[-0.5]], [["NaN"],[-0.6000000238418579]], false);
});
test("xlogy([0.30000001192092896], [0])", async () => {
    await f("xlogy", [[0.30000001192092896],[0]], [["-Inf"]]);
});
test("xlogy([0.30000001192092896], [0]) gradient", async () => {
    await b("xlogy", [[0.30000001192092896],[0]], [["-Inf"],["+Inf"]], false);
});
test("xlogy([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("xlogy", [[0.30000001192092896],[0.30000001192092896]], [[-0.36119186878204346]]);
});
test("xlogy([0.30000001192092896], [0.30000001192092896]) gradient", async () => {
    await b("xlogy", [[0.30000001192092896],[0.30000001192092896]], [[-1.2039728164672852],[1]], false);
});
test("xlogy_([-0.5], [-0.5])", async () => {
    await f("xlogy_", [[-0.5],[-0.5]], [["NaN"]]);
});
test("xlogy_([-0.5], [0])", async () => {
    await f("xlogy_", [[-0.5],[0]], [["+Inf"]]);
});
test("xlogy_([-0.5], [0.30000001192092896])", async () => {
    await f("xlogy_", [[-0.5],[0.30000001192092896]], [[0.6019864082336426]]);
});
test("xlogy_([0], [-0.5])", async () => {
    await f("xlogy_", [[0],[-0.5]], [[0]]);
});
test("xlogy_([0], [0])", async () => {
    await f("xlogy_", [[0],[0]], [[0]]);
});
test("xlogy_([0], [0.30000001192092896])", async () => {
    await f("xlogy_", [[0],[0.30000001192092896]], [[0]]);
});
test("xlogy_([0.30000001192092896], [-0.5])", async () => {
    await f("xlogy_", [[0.30000001192092896],[-0.5]], [["NaN"]]);
});
test("xlogy_([0.30000001192092896], [0])", async () => {
    await f("xlogy_", [[0.30000001192092896],[0]], [["-Inf"]]);
});
test("xlogy_([0.30000001192092896], [0.30000001192092896])", async () => {
    await f("xlogy_", [[0.30000001192092896],[0.30000001192092896]], [[-0.36119186878204346]]);
});