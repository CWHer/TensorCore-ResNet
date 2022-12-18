#pragma once

#include "common.h"
#include "simulator.hpp"

namespace Sim
{

    void wmma_kernel(f16 *a, f16 *b, f32 *c){
        // NOTE: wmma functions
        // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        // wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

        // wmma::fill_fragment(c_frag, 0.0f);
        // wmma::load_matrix_sync(a_frag, a, 16);
        // wmma::load_matrix_sync(b_frag, b, 16);
        // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        // wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);

        // TODO:
    };

    void device_gemm(Sim::GPUSimulator &sim,
                     const Sim::GPUSimulator::ThreadWarp &warp,
                     const f32 *a, const f32 *b, f32 *c, int m, int n, int k)
    {
        for (const auto &thread_num : warp)
        {
            Sim::unit3 thread_idx;
            Sim::dim3 block_dim;
            std::tie(thread_idx, block_dim) = sim.readThreadInfo(thread_num);

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
                        a_frag[i][j].fromFloat(a[(row + i) * k + col + j]);
                        b_frag[i][j].fromFloat(b[(row + i) * k + col + j]);
                        c_frag[i][j] = c[(row + i) * k + col + j];
                    }
            wmma_kernel((f16 *)a_frag, (f16 *)b_frag, (f32 *)c_frag);

            for (int i = 0; i < N_WMMA; i++)
                for (int j = 0; j < N_WMMA; j++)
                    if (row + i < m && col + j < n)
                        c[(row + i) * k + col + j] = c_frag[i][j];
        }
    }

    void host_gemm(const f32 *a, const f32 *b, f32 *c,
                   int m, int n, int k, Sim::GPUSimulator &sim)
    {
        f32 *d_a, *d_b, *d_c;
        sim.cudaMalloc((void **)&d_a, m * k * sizeof(f32));
        sim.cudaMalloc((void **)&d_b, k * n * sizeof(f32));
        sim.cudaMalloc((void **)&d_c, m * n * sizeof(f32));

        sim.cudaMemcpy(d_a, (void *)a, m * k * sizeof(f32), CUDAMemcpyType::MemcpyDeviceToHost);
        sim.cudaMemcpy(d_b, (void *)b, k * n * sizeof(f32), CUDAMemcpyType::MemcpyDeviceToHost);

        static const int N_WMMA = 16;
        // TODO: padding
        dim3 block_dim(16, 16, 32);
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