#include "tensor_core.h"

__half __float2half(const float &a)
{
    // Convert floating-point numbers
    // to half-precision floating-point numbers
}

float __half2float(const __half &a)
{
    // Convert half-precision floating-point numbers
    // to  floating-point numbers
}

float operator*(const __half &lh, const __half &rh)
{
    // Overloaded half-precision
    // floating-point number multiplication
}

GPU::GPU()
{
    // Initialize GPU resources reasonably, including regfile size and global
    // memory size, assuming sm=1 and warp_num=1
}

void GPU::SIM_LDG_INSTR()
{
    // LDG implementation
}
void GPU::SIM_STG_INSTR()
{
    // STG implementation
}
void GPU::SIM_HMMA_INSTR_STEP0()
{
    // HMMA.STEP0 implementation
}
void GPU::SIM_HMMA_INSTR_STEP1()
{
    // HMMA.STEP1 implementation
}
void GPU::SIM_HMMA_INSTR_STEP2()
{
    // HMMA.STEP2 implementation
}
void GPU::SIM_HMMA_INSTR_STEP3()
{
    // HMMA.STEP3 implementation
}

void GPU::SIM_S2R_INSTR()
{
    // S2R implementation
}
void GPU::SIM_IMAD_INSTR()
{
    // IMAD implementation
}

void GPU::SIM_LOP3_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
                         unsigned imm)
{
    // for: warp execuation
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++)
    {
        // LOP3 implementation
        unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
        unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
        unsigned &sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
        unsigned data = 0;
        if (imm & 0x01)
            data |= (~ra_data) & (~sb_data) & (~sc_data);
        if (imm & 0x02)
            data |= (~ra_data) & (~sb_data) & (sc_data);
        if (imm & 0x04)
            data |= (~ra_data) & (sb_data) & (~sc_data);
        if (imm & 0x08)
            data |= (~ra_data) & (sb_data) & (sc_data);
        if (imm & 0x10)
            data |= (ra_data) & (~sb_data) & (~sc_data);
        if (imm & 0x20)
            data |= (ra_data) & (~sb_data) & (sc_data);
        if (imm & 0x40)
            data |= (ra_data) & (sb_data) & (~sc_data);
        if (imm & 0x80)
            data |= (ra_data) & (sb_data) & (sc_data);
        regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
    }
}

void GPU::SIM_SHF_INSTR()
{
    // SHF implementation
}

void GPU::SIM_CS2R_INSTR()
{
    // S2R implementation
}

void GPU::SIM_LEA_INSTR(bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb,
                        unsigned imm, unsigned Pd0, unsigned Ps0)
{
    // for: warp execuation
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++)
    {
        // LEA implementation
        unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
        unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
        uint64_t data = ra_data;
        if (HI)
            data = data >> (32 - imm);
        else
            data = data << imm;
        data += sb_data;
        if (X)
            data += pregfile_[Ps0 * WARP_SIZE_ + threadIdx];
        if (Pd0 != 7)
            pregfile_[Pd0 * WARP_SIZE_ + threadIdx] = ((data >> 32) & 0x1);
        data &= 0xffffffff;
        regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
    }
}

void GPU::SIM_EXIT_INSTR()
{
    // EXIT implementation
}

void simMalloc(void **ptr, size_t size, GPU &volta)
{
    // sim cudaMalloc
    // Request GPU memory
}

void simMemcpy(void *dst, void *src, size_t count, enum cudaMemcpyKind kind,
               GPU &volta)
{
    // sim cudaMemcpy
    // memcpy host memory to class GPU memory or
    // memcpy class GPU memory to host memory
}

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
