#pragma once

#include "common.h"

namespace Sim
{
    // TODO: migrate original code here

    void wmma_kernel(__half *a, __half *b, float *c, float *d, dim3 &gridDim,
                     dim3 &blockDim, GPU &volta)
    {
        // device kernel function
        // gridDim & blockDim
        // assume c[0x0][0x28]=0
        const int c_0_28 = 0;
        volta.SIM_IMAD_INSTR(); // SASS: IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
                                // add instruction you need,sim_imad_instr() is just an example
    }

    void gemm(float *a, float *b, float *c, float *d)
    {
        // host function gemm
        // mxnxk=? According to the pytorch source code, find the matrix size of the
        // conv2d operator of Resnet18 when doing matrix multiplication after
        // im2col. Remember to turn off cudnn and use the cublas library We need to
        // considerthe settings of blockDim and gridDim, we limit only one warp, how
        // to slicethe matrix to a matrix of 16*16*16 size, what to do if it cannot
        // be divisible evenly
    }

    void im2col()
    {
        // ref caffe
    }
    void conv() {}

}