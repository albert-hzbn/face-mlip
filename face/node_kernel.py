"""Fused node-level invariant algebra (NVRTC via CuPy).

One thread per (atom, channel): the 8 polynomial invariant families (3-, 4-, 5-body) and the 8 gated-product
paths are evaluated from P1,Q1,S1 (3), P2,Q2,S2 (3x3), P3,Q3 (3x3x3) held in registers. The PyTorch version
writes a full (N,C,3,3,3[,3]) temporary per broadcast op (~40 % of inference time). The backward is the
hand-derived product rule (first order: inference / MD), verified against autograd in tests/test_kernels.py.
"""
import torch

from .kernels import cp, _cp

_NODE_SRC = r'''
#define I2(a,b) ((a)*3+(b))
#define I3(a,b,c) ((a)*9+(b)*3+(c))
// full index -> symmetric component (face.cartesian.SYM_IDX order)
__constant__ int E2[9]  = {0,3,4, 3,1,5, 4,5,2};
__constant__ int E3[27] = {0,3,4, 3,5,9, 4,9,7,   3,5,9, 5,1,6, 9,6,8,   4,9,7, 9,6,8, 7,8,2};
extern "C" __global__ void node_fwd(const float* __restrict__ P1, const float* __restrict__ Q1, const float* __restrict__ S1,
    const float* __restrict__ P2, const float* __restrict__ Q2, const float* __restrict__ S2,
    const float* __restrict__ P3, const float* __restrict__ Q3,
    const float* __restrict__ av, const float* __restrict__ bm,
    float* __restrict__ inv, float* __restrict__ V, float* __restrict__ Mf, const long NC, const int C, const int eq) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= NC) return;
    int c = t % C;
    float p1[3], q1[3], s1[3], p2[9], q2[9], s2[9], p3[27], q3[27];
    for (int k = 0; k < 3; k++) { p1[k] = P1[t*3+k]; q1[k] = Q1[t*3+k]; s1[k] = S1[t*3+k]; }
    for (int k = 0; k < 9; k++) { p2[k] = P2[t*6+E2[k]]; q2[k] = Q2[t*6+E2[k]]; s2[k] = S2[t*6+E2[k]]; }
    for (int k = 0; k < 27; k++) { p3[k] = P3[t*10+E3[k]]; q3[k] = Q3[t*10+E3[k]]; }
    float i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0, i8 = 0;
    for (int a = 0; a < 3; a++) i1 += p1[a] * q1[a];
    for (int k = 0; k < 9; k++) i2 += p2[k] * q2[k];
    for (int k = 0; k < 27; k++) i3 += p3[k] * q3[k];
    float q2s1[3], p3q2[3], p2q1[3], pq2[9], p3s1[9];
    for (int a = 0; a < 3; a++) {
        float x = 0, y = 0, z = 0;
        for (int b = 0; b < 3; b++) { x += q2[I2(a,b)] * s1[b]; z += p2[I2(a,b)] * q1[b]; }
        for (int b = 0; b < 3; b++) for (int cc = 0; cc < 3; cc++) y += p3[I3(a,b,cc)] * q2[I2(b,cc)];
        q2s1[a] = x; p3q2[a] = y; p2q1[a] = z;
    }
    for (int a = 0; a < 3; a++) for (int d = 0; d < 3; d++) {
        float x = 0, y = 0;
        for (int b = 0; b < 3; b++) { x += p2[I2(a,b)] * q2[I2(b,d)]; y += p3[I3(a,d,b)] * s1[b]; }
        pq2[I2(a,d)] = x; p3s1[I2(a,d)] = y;
    }
    for (int a = 0; a < 3; a++) { i4 += p1[a] * q2s1[a]; i6 += p3q2[a] * s1[a]; }
    for (int a = 0; a < 3; a++) for (int d = 0; d < 3; d++) i5 += pq2[I2(a,d)] * s2[I2(d,a)];
    for (int d = 0; d < 3; d++) for (int e = 0; e < 3; e++) {
        float x = 0;
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) x += p3[I3(a,b,d)] * q3[I3(a,b,e)];
        i7 += x * s2[I2(d,e)];
    }
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int d = 0; d < 3; d++)
        i8 += p3[I3(a,b,d)] * q1[a] * s1[b] * q1[d];
    float* o = inv + t * 8;
    o[0] = i1; o[1] = i2; o[2] = i3; o[3] = i4; o[4] = i5; o[5] = i6; o[6] = i7; o[7] = i8;
    if (eq) {
        const float a0 = av[c*4], a1 = av[c*4+1], a2 = av[c*4+2], a3 = av[c*4+3];
        const float b0 = bm[c*4], b1 = bm[c*4+1], b2 = bm[c*4+2], b3 = bm[c*4+3];
        for (int a = 0; a < 3; a++) V[t*3+a] = a0*p1[a] + a1*q2s1[a] + a2*p3q2[a] + a3*p2q1[a];
        for (int a = 0; a < 3; a++) for (int d = 0; d < 3; d++)
            Mf[t*9+I2(a,d)] = b0*p2[I2(a,d)] + b1*p1[a]*q1[d] + b2*pq2[I2(a,d)] + b3*p3s1[I2(a,d)];
    }
}

extern "C" __global__ void node_bwd(const float* __restrict__ P1, const float* __restrict__ Q1, const float* __restrict__ S1,
    const float* __restrict__ P2, const float* __restrict__ Q2, const float* __restrict__ S2,
    const float* __restrict__ P3, const float* __restrict__ Q3,
    const float* __restrict__ av, const float* __restrict__ bm,
    const float* __restrict__ gI, const float* __restrict__ gV, const float* __restrict__ gM,
    float* dP1, float* dQ1, float* dS1, float* dP2, float* dQ2, float* dS2, float* dP3, float* dQ3,
    float* da, float* db, const long NC, const int C, const int eq) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= NC) return;
    int c = t % C;
    float p1[3], q1[3], s1[3], p2[9], q2[9], s2[9];
    for (int k = 0; k < 3; k++) { p1[k] = P1[t*3+k]; q1[k] = Q1[t*3+k]; s1[k] = S1[t*3+k]; }
    for (int k = 0; k < 9; k++) { p2[k] = P2[t*6+E2[k]]; q2[k] = Q2[t*6+E2[k]]; s2[k] = S2[t*6+E2[k]]; }
    float p3[27], q3[27], gp3[10], gq3[10];
    for (int k = 0; k < 27; k++) { p3[k] = P3[t*10+E3[k]]; q3[k] = Q3[t*10+E3[k]]; }
    for (int k = 0; k < 10; k++) { gp3[k] = 0; gq3[k] = 0; }
    const float g1 = gI[t*8], g2 = gI[t*8+1], g3 = gI[t*8+2], g4 = gI[t*8+3];
    const float g5 = gI[t*8+4], g6 = gI[t*8+5], g7 = gI[t*8+6], g8 = gI[t*8+7];
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0, gv[3] = {0, 0, 0}, gm[9];
    for (int k = 0; k < 9; k++) gm[k] = 0;
    if (eq) {
        a0 = av[c*4]; a1 = av[c*4+1]; a2 = av[c*4+2]; a3 = av[c*4+3];
        b0 = bm[c*4]; b1 = bm[c*4+1]; b2 = bm[c*4+2]; b3 = bm[c*4+3];
        for (int k = 0; k < 3; k++) gv[k] = gV[t*3+k];
        for (int k = 0; k < 9; k++) gm[k] = gM[t*9+k];
    }
    float dp1[3], dq1[3], ds1[3], dp2[9], dq2[9], ds2[9];
    for (int a = 0; a < 3; a++) { dp1[a] = g1 * q1[a] + a0 * gv[a]; dq1[a] = g1 * p1[a]; ds1[a] = 0; }
    for (int k = 0; k < 9; k++) { dp2[k] = g2 * q2[k] + b0 * gm[k]; dq2[k] = g2 * p2[k]; ds2[k] = 0; }
    float da0 = 0, da1 = 0, da2 = 0, da3 = 0, db0 = 0, db1 = 0, db2 = 0, db3 = 0;
    for (int a = 0; a < 3; a++) da0 += gv[a] * p1[a];
    for (int k = 0; k < 9; k++) db0 += gm[k] * p2[k];
    // inv4 = P1a Q2ab S1b ; V a1: Q2ab S1b ; V a3: P2ab Q1b ; M b1: P1a Q1b
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        dp1[a] += g4 * q2[I2(a,b)] * s1[b];
        dq2[I2(a,b)] += (g4 * p1[a] + a1 * gv[a]) * s1[b];
        ds1[b] += (g4 * p1[a] + a1 * gv[a]) * q2[I2(a,b)];
        da1 += gv[a] * q2[I2(a,b)] * s1[b];
        dp2[I2(a,b)] += a3 * gv[a] * q1[b];
        dq1[b] += a3 * gv[a] * p2[I2(a,b)];
        da3 += gv[a] * p2[I2(a,b)] * q1[b];
        dp1[a] += b1 * gm[I2(a,b)] * q1[b];
        dq1[b] += b1 * gm[I2(a,b)] * p1[a];
        db1 += gm[I2(a,b)] * p1[a] * q1[b];
    }
    // inv5 = P2ab Q2bd S2da ; M b2: PQ2ad = P2ab Q2bd
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int d = 0; d < 3; d++) {
        const float w = g5 * s2[I2(d,a)] + b2 * gm[I2(a,d)];
        dp2[I2(a,b)] += w * q2[I2(b,d)];
        dq2[I2(b,d)] += w * p2[I2(a,b)];
        ds2[I2(d,a)] += g5 * p2[I2(a,b)] * q2[I2(b,d)];
        db2 += gm[I2(a,d)] * p2[I2(a,b)] * q2[I2(b,d)];
    }
    // rank-3 terms: inv3, inv6, inv7, inv8, V a2, M b3
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int d = 0; d < 3; d++) {
        const int k = I3(a,b,d);
        const float pv = p3[k];
        float s2q3 = 0, p3s2 = 0;
        for (int e = 0; e < 3; e++) { s2q3 += q3[I3(a,b,e)] * s2[I2(d,e)]; p3s2 += p3[I3(a,b,e)] * s2[I2(e,d)]; }
        gp3[E3[k]] += g3 * q3[k] + g6 * q2[I2(b,d)] * s1[a] + g7 * s2q3 + g8 * q1[a] * s1[b] * q1[d]
                    + a2 * gv[a] * q2[I2(b,d)] + b3 * gm[I2(a,b)] * s1[d];
        gq3[E3[k]] += g3 * pv + g7 * p3s2;
        dq2[I2(b,d)] += (g6 * s1[a] + a2 * gv[a]) * pv;
        ds1[a] += g6 * pv * q2[I2(b,d)];
        da2 += gv[a] * pv * q2[I2(b,d)];
        dq1[a] += g8 * pv * s1[b] * q1[d];
        dq1[d] += g8 * pv * q1[a] * s1[b];
        ds1[b] += g8 * pv * q1[a] * q1[d];
        ds1[d] += b3 * gm[I2(a,b)] * pv;
        db3 += gm[I2(a,b)] * pv * s1[d];
    }
    for (int d = 0; d < 3; d++) for (int e = 0; e < 3; e++) {
        float x = 0;
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) x += p3[I3(a,b,d)] * q3[I3(a,b,e)];
        ds2[I2(d,e)] += g7 * x;
    }
    for (int k = 0; k < 3; k++) { dP1[t*3+k] = dp1[k]; dQ1[t*3+k] = dq1[k]; dS1[t*3+k] = ds1[k]; }
    float o2p[6] = {0,0,0,0,0,0}, o2q[6] = {0,0,0,0,0,0}, o2s[6] = {0,0,0,0,0,0};
    for (int k = 0; k < 9; k++) { o2p[E2[k]] += dp2[k]; o2q[E2[k]] += dq2[k]; o2s[E2[k]] += ds2[k]; }
    for (int k = 0; k < 6; k++) { dP2[t*6+k] = o2p[k]; dQ2[t*6+k] = o2q[k]; dS2[t*6+k] = o2s[k]; }
    for (int k = 0; k < 10; k++) { dP3[t*10+k] = gp3[k]; dQ3[t*10+k] = gq3[k]; }
    if (eq) {
        da[t*4] = da0; da[t*4+1] = da1; da[t*4+2] = da2; da[t*4+3] = da3;
        db[t*4] = db0; db[t*4+1] = db1; db[t*4+2] = db2; db[t*4+3] = db3;
    }
}
'''

_mod = None


def _kern(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_NODE_SRC, options=("--use_fast_math",))
    return _mod.get_function(name)


class NodeAlgebra(torch.autograd.Function):
    """Symmetric storage in: (P1,Q1,S1 (N,C,3); P2,Q2,S2 (N,C,6); P3,Q3 (N,C,10); a,b (4,C)) ->
       inv (N,C,8) [families 1..8], V (N,C,3), Mf (N,C,3,3)."""

    @staticmethod
    def forward(ctx, P1, Q1, S1, P2, Q2, S2, P3, Q3, a, b, eq):
        N, C = P1.shape[:2]
        NC = N * C
        ins = [x.contiguous().float() for x in (P1, Q1, S1, P2, Q2, S2, P3, Q3)]
        at, bt = a.t().contiguous().float(), b.t().contiguous().float()          # (C,4)
        inv = torch.empty(N, C, 8, device=P1.device)
        V = torch.empty(N, C, 3, device=P1.device)
        Mf = torch.empty(N, C, 3, 3, device=P1.device)
        thr = 128
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            _kern("node_fwd")(((NC + thr - 1) // thr,), (thr,),
                              (*[_cp(x) for x in ins], _cp(at), _cp(bt), _cp(inv), _cp(V), _cp(Mf),
                               cp.int64(NC), cp.int32(C), cp.int32(int(eq))))
        ctx.save_for_backward(*ins, at, bt)
        ctx.eq = eq
        return inv, V, Mf

    @staticmethod
    def backward(ctx, gI, gV, gM):
        *ins, at, bt = ctx.saved_tensors
        N, C = ins[0].shape[:2]
        NC = N * C
        dev = ins[0].device
        gI = (gI if gI is not None else torch.zeros(N, C, 8, device=dev)).contiguous().float()
        gV = (gV if gV is not None else torch.zeros(N, C, 3, device=dev)).contiguous().float()
        gM = (gM if gM is not None else torch.zeros(N, C, 3, 3, device=dev)).contiguous().float()
        outs = [torch.empty_like(x) for x in ins]
        da = torch.zeros(N, C, 4, device=dev)
        db = torch.zeros(N, C, 4, device=dev)
        thr = 128
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            _kern("node_bwd")(((NC + thr - 1) // thr,), (thr,),
                              (*[_cp(x) for x in ins], _cp(at), _cp(bt), _cp(gI), _cp(gV), _cp(gM),
                               *[_cp(x) for x in outs], _cp(da), _cp(db),
                               cp.int64(NC), cp.int32(C), cp.int32(int(ctx.eq))))
        return (*outs, da.sum(0).t(), db.sum(0).t(), None)


def node_algebra(P1, Q1, S1, P2, Q2, S2, P3, Q3, a, b, eq=True):
    """P2,Q2,S2: (N,C,6) and P3,Q3: (N,C,10) symmetric components (no expansion to full tensors needed)."""
    return NodeAlgebra.apply(P1, Q1, S1, P2, Q2, S2, P3, Q3, a, b, eq)


def node_algebra_ref(P1, Q1, S1, P2, Q2, S2, P3, Q3, a, b):
    """PyTorch reference (same formulas as MomentBlock.invariants / equivariant_out)."""
    PQ2 = (P2[..., :, :, None] * Q2[..., None, :, :]).sum(-2)
    X33 = (P3[..., :, :, :, None] * Q3[..., :, :, None, :]).sum((2, 3))
    inv = torch.stack([
        (P1 * Q1).sum(-1), (P2 * Q2).sum((-1, -2)), (P3 * Q3).sum((-1, -2, -3)),
        ((P1[..., :, None] * Q2).sum(-2) * S1).sum(-1),
        (PQ2 * S2.transpose(-1, -2)).sum((-1, -2)),
        ((P3 * Q2[..., None, :, :]).sum((-1, -2)) * S1).sum(-1),
        (X33 * S2).sum((-1, -2)),
        (((P3 * Q1[..., None, None, :]).sum(-1) * S1[..., None, :]).sum(-1) * Q1).sum(-1)], -1)
    a, b = a[..., None], b[..., None, None]
    V = (a[0] * P1 + a[1] * (Q2 * S1[..., None, :]).sum(-1)
         + a[2] * (P3 * Q2[..., None, :, :]).sum((-1, -2)) + a[3] * (P2 * Q1[..., None, :]).sum(-1))
    Mf = (b[0] * P2 + b[1] * P1[..., :, None] * Q1[..., None, :] + b[2] * PQ2
          + b[3] * (P3 * S1[..., None, None, :]).sum(-1))
    return inv, V, Mf
