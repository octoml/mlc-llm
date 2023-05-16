from tvm.relax.dpl import PatternContext, is_const, is_op, rewrite_call, wildcard
from tvm.script import relax as R


def rewrite_attention(f):
    Q = wildcard()
    K = wildcard()
    V = wildcard()
    bias = wildcard()

    Q_BNSH = is_op("relax.permute_dims")(Q)
    K_BNSH = is_op("relax.permute_dims")(K)
    V_BNSH = is_op("relax.permute_dims")(V)

    K_BNSH_T = is_op("relax.permute_dims")(K_BNSH)

    matmul1 = is_op("relax.matmul")(Q_BNSH, K_BNSH_T)
    divide = is_op("relax.divide")(matmul1, is_const())
    with_bias = is_op("relax.add")(divide, bias)
    softmax = is_op("relax.nn.softmax")(with_bias)
    matmul2 = is_op("relax.matmul")(softmax, V_BNSH)

    pattern = is_op("relax.permute_dims")(matmul2)

    def callback(_, matchings):
        return R.nn.attention(matchings[Q], matchings[K], matchings[V], matchings[bias])

    return rewrite_call(pattern, callback, f)
