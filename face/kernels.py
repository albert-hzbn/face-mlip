"""Fused CUDA kernels (NVRTC via CuPy; no host C++ compiler needed).

factor_density:  D[i, x, b*K + k] = sum_m phi[i,m,b] * msg[i,m,k] * Y[i,m,x]       x in 0..19
  forward : one block per atom, threads over (b, k); 20 register accumulators per thread;
            neighbour tiles of phi / msg / Y staged in shared memory -> the per-edge B*K tensor
            that dominates the PyTorch implementation is never materialised.
  backward: G = dL/dD of one atom kept in shared memory; per edge T[b,k] = sum_x G[x,b,k] Y[x]
            is formed once and reused for dphi = sum_k msg T, dmsg = sum_b phi T, and
            dY[x] = sum_{b,k} G[x,b,k] phi_b msg_k.
First-order differentiable only (inference / MD forces); training uses the PyTorch path, which
computes the identical function (tests/test_kernels.py).
"""
import torch

try:
    import cupy as cp
except Exception:            # pragma: no cover
    cp = None

_SRC = r'''
#define NX 20
extern "C" __global__ void dens_fwd(const float* __restrict__ phi, const float* __restrict__ msg,
                                    const float* __restrict__ Y, float* __restrict__ D,
                                    const int M, const int B, const int K, const int TILE) {
    extern __shared__ float sh[];
    float* sphi = sh;                       // TILE*B
    float* smsg = sphi + TILE * B;          // TILE*K
    float* sY   = smsg + TILE * K;          // TILE*20
    const int i = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int BK = B * K;
    float acc[NX];
    #pragma unroll
    for (int x = 0; x < NX; x++) acc[x] = 0.f;
    const int b = tid / K, k = tid % K;
    for (int m0 = 0; m0 < M; m0 += TILE) {
        const int nm = min(TILE, M - m0);
        for (int q = tid; q < nm * B; q += nt) sphi[q] = phi[((long)i * M + m0) * B + q];
        for (int q = tid; q < nm * K; q += nt) smsg[q] = msg[((long)i * M + m0) * K + q];
        for (int q = tid; q < nm * NX; q += nt) sY[q] = Y[((long)i * M + m0) * NX + q];
        __syncthreads();
        if (tid < BK) {
            for (int t = 0; t < nm; t++) {
                const float p = sphi[t * B + b] * smsg[t * K + k];
                #pragma unroll
                for (int x = 0; x < NX; x++) acc[x] += p * sY[t * NX + x];
            }
        }
        __syncthreads();
    }
    if (tid < BK) {
        #pragma unroll
        for (int x = 0; x < NX; x++) D[((long)i * NX + x) * BK + tid] = acc[x];
    }
}

extern "C" __global__ void dens_bwd(const float* __restrict__ phi, const float* __restrict__ msg,
                                    const float* __restrict__ Y, const float* __restrict__ G,
                                    float* __restrict__ dphi, float* __restrict__ dmsg,
                                    float* __restrict__ dY, const int M, const int B, const int K,
                                    const int TILE) {
    // shared rows padded to K+1 floats: b-strided accesses are otherwise 32-way bank conflicts
    extern __shared__ float sh[];
    const int BK = B * K, PK = K + 1, BPK = B * PK;
    float* sG  = sh;                        // 20*BPK
    float* sT  = sG + NX * BPK;             // TILE*BPK
    float* sp  = sT + TILE * BPK;           // TILE*B
    float* sm  = sp + TILE * B;             // TILE*K
    float* sy  = sm + TILE * K;             // TILE*20
    float* sdY = sy + TILE * NX;            // TILE*20
    const int i = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    for (int q = tid; q < NX * BK; q += nt) {
        const int x = q / BK, c = q - x * BK, bb = c / K, k = c - bb * K;
        sG[x * BPK + bb * PK + k] = G[(long)i * NX * BK + q];
    }
    for (int m0 = 0; m0 < M; m0 += TILE) {
        const int nm = min(TILE, M - m0);
        const long e0 = (long)i * M + m0;
        for (int q = tid; q < nm * B; q += nt) sp[q] = phi[e0 * B + q];
        for (int q = tid; q < nm * K; q += nt) sm[q] = msg[e0 * K + q];
        for (int q = tid; q < nm * NX; q += nt) { sy[q] = Y[e0 * NX + q]; sdY[q] = 0.f; }
        __syncthreads();
        for (int q = tid; q < nm * BK; q += nt) {          // T[t,b,k] = sum_x G[x,b,k] Y[t,x]
            const int t = q / BK, c = q - t * BK, bb = c / K, k = c - bb * K;
            const int o = bb * PK + k;
            float s = 0.f;
            #pragma unroll
            for (int x = 0; x < NX; x++) s += sG[x * BPK + o] * sy[t * NX + x];
            sT[t * BPK + o] = s;
        }
        __syncthreads();
        const int nA = nm * (B + K), nB = nm * NX * B;
        for (int q = tid; q < nA + nB; q += nt) {
            if (q < nm * B) {
                const int t = q / B, bb = q - t * B;
                const float* T = sT + t * BPK + bb * PK;
                const float* mm = sm + t * K;
                float s = 0.f;
                for (int k = 0; k < K; k++) s += mm[k] * T[k];
                dphi[(e0 + t) * B + bb] = s;
            } else if (q < nA) {
                const int r = q - nm * B, t = r / K, k = r - t * K;
                float s = 0.f;
                for (int bb = 0; bb < B; bb++) s += sp[t * B + bb] * sT[t * BPK + bb * PK + k];
                dmsg[(e0 + t) * K + k] = s;
            } else {
                const int r = q - nA, t = r / (NX * B), rem = r - t * NX * B, x = rem / B, bb = rem - x * B;
                const float* g = sG + x * BPK + bb * PK;
                const float* mm = sm + t * K;
                float s = 0.f;
                for (int k = 0; k < K; k++) s += g[k] * mm[k];
                atomicAdd(&sdY[t * NX + x], s * sp[t * B + bb]);
            }
        }
        __syncthreads();
        for (int q = tid; q < nm * NX; q += nt) dY[e0 * NX + q] = sdY[q];
        __syncthreads();
    }
}
'''

_mod = None


def _kern(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, options=("--use_fast_math",))
    return _mod.get_function(name)


def _cp(t):
    return cp.from_dlpack(t.detach().contiguous())


class FactorDensity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phi, msg, Y):
        N, M, B = phi.shape
        K = msg.shape[-1]
        phi, msg, Y = phi.contiguous().float(), msg.contiguous().float(), Y.contiguous().float()
        D = torch.empty(N, 20, B * K, device=phi.device, dtype=torch.float32)
        TILE = 16
        f = _kern("dens_fwd")
        smem = 4 * TILE * (B + K + 20)
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            f((N,), (max(B * K, 32),), (_cp(phi), _cp(msg), _cp(Y), _cp(D), cp.int32(M), cp.int32(B),
                                         cp.int32(K), cp.int32(TILE)), shared_mem=smem)
        ctx.save_for_backward(phi, msg, Y)
        return D

    @staticmethod
    def backward(ctx, G):
        phi, msg, Y = ctx.saved_tensors
        N, M, B = phi.shape
        K = msg.shape[-1]
        G = G.contiguous().float()
        dphi, dmsg, dY = torch.empty_like(phi), torch.empty_like(msg), torch.empty_like(Y)
        f = _kern("dens_bwd")
        BPK = B * (K + 1)
        TILE = max(1, min(16, (96 * 1024 // 4 - 20 * BPK) // (BPK + B + K + 40)))
        smem = 4 * (20 * BPK + TILE * (BPK + B + K + 40))
        if smem > 48 * 1024:
            f.max_dynamic_shared_size_bytes = smem
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            f((N,), (512,), (_cp(phi), _cp(msg), _cp(Y), _cp(G), _cp(dphi), _cp(dmsg), _cp(dY),
                             cp.int32(M), cp.int32(B), cp.int32(K), cp.int32(TILE)), shared_mem=smem)
        return dphi, dmsg, dY


def factor_density(phi, msg, Y):
    """(N,M,B), (N,M,K), (N,M,20) -> (N,20,B*K); first-order differentiable."""
    return FactorDensity.apply(phi, msg, Y)


def factor_density_ref(phi, msg, Y):
    N, M, B = phi.shape
    w = (phi[..., :, None] * msg[..., None, :]).reshape(N, M, -1)
    return torch.bmm(Y.transpose(1, 2), w)
