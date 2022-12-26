#include "functions.h"

namespace Sim
{

    void wmma_kernel(GPUSimulator &sim,
                     const GPUSimulator::ThreadWarp &warp,
                     f16 *a, f16 *b, f32 *c)
    {
        sim.S2R_INSTR(warp, 2, SRegisterType::SR_LANEID);
        sim.IMAD_INSTR(warp, false, 12, 255, 255, 0x10, 0b001);
        sim.SHF_INSTR(warp, false, 3, 2, 0x2, 255);
        sim.LOP3_INSTR(warp, 29, 2, 0x3, 255, 0xc0, 0b010);
        sim.LOP3_INSTR(warp, 3, 3, 0x3, 255, 0xc0, 0b010);
        sim.SHF_INSTR(warp, false, 2, 2, 0x4, 255);
        sim.SHF_INSTR(warp, false, 0, 3, 0x1, 255);
        sim.IMAD_INSTR(warp, false, 6, 3, 0x8, 255, 0b010);
        sim.LOP3_INSTR(warp, 4, 2, 0x1, 255, 0xc0, 0b010);
        sim.IMAD_INSTR(warp, false, 7, 0, 0x8, 29, 0b010);
        sim.LOP3_INSTR(warp, 5, 6, 0x8, 29, 0xe2, 0b010);
        sim.IMAD_INSTR(warp, false, 7, 4, 0x4, 7, 0b010);
        sim.IMAD_INSTR(warp, false, 5, 4, 0x4, 5, 0b010);
        sim.IMAD_INSTR(warp, false, 7, 7, 0x2, 255, 0b010);
        sim.SHF_INSTR(warp, false, 5, 5, 0x1, 255);
        sim.IMAD_INSTR(warp, true, 20, 5, 12, reinterpret_cast<intptr_t>(a), 0b001);
        sim.IMAD_INSTR(warp, true, 12, 7, 12, reinterpret_cast<intptr_t>(b), 0b001);
        sim.LDG_INSTR(warp, 128, 24, 20, 0x0);
        sim.LDG_INSTR(warp, 128, 16, 12, 0x0);
        sim.LDG_INSTR(warp, 128, 20, 20, 0x10);
        sim.LDG_INSTR(warp, 128, 12, 12, 0x10);
        // TODO
    };

    void device_gemm(GPUSimulator &sim,
                     const GPUSimulator::ThreadWarp &warp,
                     const f32 *a, const f32 *b, f32 *c, int m, int n, int k)
    {
        int thread_num = warp.front();
        unit3 thread_idx;
        dim3 block_dim;
        std::tie(thread_idx, block_dim) = sim.readThreadInfo(thread_num);

        // NOTE: HACK: host_gemm() set block_dim.z = WARP_SIZE
        for (const auto &thread_num : warp)
        {
            unit3 __thread_idx;
            std::tie(__thread_idx, std::ignore) = sim.readThreadInfo(thread_num);
            printCppError(__thread_idx.x != thread_idx.x ||
                              __thread_idx.y != thread_idx.y,
                          "set block_dim.z == WARP_SIZE should ensure all threads in a warp \
                           have the same thread_idx.x and thread_idx.y",
                          __FILE__, __LINE__);
        }

        static const int N_WMMA = 16;

        f16 a_frag[N_WMMA][N_WMMA], b_frag[N_WMMA][N_WMMA];
        f32 c_frag[N_WMMA][N_WMMA];

        int row = thread_idx.y * N_WMMA;
        int col = thread_idx.x * N_WMMA;
        for (int i = 0; i < N_WMMA; i++)
            for (int j = 0; j < N_WMMA; j++)
                if (row + i < m && col + j < n)
                {
                    // TODO: check this
                    a_frag[i][j] = __float2half(a[(row + i) * k + col + j]);
                    b_frag[i][j] = __float2half(b[(row + i) * k + col + j]);
                    c_frag[i][j] = c[(row + i) * k + col + j];
                }
        wmma_kernel(sim, warp, (f16 *)a_frag, (f16 *)b_frag, (f32 *)c_frag);

        for (int i = 0; i < N_WMMA; i++)
            for (int j = 0; j < N_WMMA; j++)
                if (row + i < m && col + j < n)
                    c[(row + i) * k + col + j] = c_frag[i][j];
    }

    void host_gemm(const f32 *a, const f32 *b, f32 *c,
                   int m, int n, int k, GPUSimulator &sim)
    {
        f32 *d_a, *d_b, *d_c;
        sim.cudaMalloc((void **)&d_a, m * k * sizeof(f32));
        sim.cudaMalloc((void **)&d_b, k * n * sizeof(f32));
        sim.cudaMalloc((void **)&d_c, m * n * sizeof(f32));

        sim.cudaMemcpy(d_a, (void *)a, m * k * sizeof(f32), CUDAMemcpyType::MemcpyDeviceToHost);
        sim.cudaMemcpy(d_b, (void *)b, k * n * sizeof(f32), CUDAMemcpyType::MemcpyDeviceToHost);

        static const int N_WMMA = 16;
        // TODO: padding
        dim3 block_dim(1, 1, GPUSimulator::WARP_SIZE);
        sim.launchKernel(block_dim, device_gemm, d_a, d_b, d_c, m, n, k);

        sim.cudaMemcpy((void *)c, d_c, m * n * sizeof(f32), CUDAMemcpyType::MemcpyDeviceToHost);
        sim.cudaFree(d_a);
        sim.cudaFree(d_b);
    }

    // void wmma_kernel(__half *a, __half *b, float *c, float *d, dim3 &gridDim,
    //                  dim3 &blockDim, GPU &volta)
    // {
    //     // device kernel function
    //     // gridDim & blockDim
    //     // assume c[0x0][0x28]=0
    //     const int c_0_28 = 0;
    //     volta.SIM_IMAD_INSTR(); // SASS: IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
    //                             // add instruction you need,sim_imad_instr() is just an example
    // }

    // void gemm(float *a, float *b, float *c, float *d)
    // {
    //     // host function gemm
    //     // mxnxk=? According to the pytorch source code, find the matrix size of the
    //     // conv2d operator of Resnet18 when doing matrix multiplication after
    //     // im2col. Remember to turn off cudnn and use the cublas library We need to
    //     // considerthe settings of blockDim and gridDim, we limit only one warp, how
    //     // to slicethe matrix to a matrix of 16*16*16 size, what to do if it cannot
    //     // be divisible evenly
    // }

    // void im2col()
    // {
    //     // ref caffe
    // }
    // void conv() {}

}