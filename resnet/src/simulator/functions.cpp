#include "functions.hpp"

namespace Sim {

// NOTE: C = A * B.T
void wmmaKernel(GPUSimulator &sim,
                const GPUSimulator::ThreadWarp &warp,
                f16 *a, f16 *b, f32 *c) {
  sim.S2R_INSTR(warp, 2, SRegisterType::SR_LANEID);                            // S2R R2, SR_LANEID ;
  sim.IMAD_INSTR(warp, false, 12, 255, 255, 0x10, 0b001);                      // IMAD.MOV.U32 R12, RZ, RZ, 0x10 ;
  sim.SHF_INSTR(warp, false, 3, 2, 0x2, 255);                                  // SHF.R.U32.HI R3, RZ, 0x2, R2.reuse ;
  sim.LOP3_INSTR(warp, 29, 2, 0x3, 255, 0xc0, 0b010);                          // LOP3.LUT R29, R2, 0x3, RZ, 0xc0, !PT ;
  sim.LOP3_INSTR(warp, 3, 3, 0x3, 255, 0xc0, 0b010);                           // LOP3.LUT R3, R3, 0x3, RZ, 0xc0, !PT ;
  sim.SHF_INSTR(warp, false, 2, 2, 0x4, 255);                                  // SHF.R.U32.HI R2, RZ, 0x4, R2 ;
  sim.SHF_INSTR(warp, false, 0, 3, 0x1, 255);                                  // SHF.R.U32.HI R0, RZ, 0x1, R3 ;
  sim.IMAD_INSTR(warp, false, 6, 3, 0x8, 255, 0b010);                          // IMAD.SHL.U32 R6, R3, 0x8, RZ ;
  sim.LOP3_INSTR(warp, 4, 2, 0x1, 255, 0xc0, 0b010);                           // LOP3.LUT R4, R2, 0x1, RZ, 0xc0, !PT ;
  sim.IMAD_INSTR(warp, false, 7, 0, 0x8, 29, 0b010);                           // IMAD R7, R0, 0x8, R29.reuse ;
  sim.LOP3_INSTR(warp, 5, 6, 0x8, 29, 0xe2, 0b010);                            // LOP3.LUT R5, R6, 0x8, R29, 0xe2, !PT ;
  sim.IMAD_INSTR(warp, false, 7, 4, 0x4, 7, 0b010);                            // IMAD R7, R4.reuse, 0x4, R7 ;
  sim.IMAD_INSTR(warp, false, 5, 4, 0x4, 5, 0b010);                            // IMAD R5, R4, 0x4, R5 ;
  sim.IMAD_INSTR(warp, false, 7, 7, 0x2, 255, 0b010);                          // IMAD.SHL.U32 R7, R7, 0x2, RZ ;
  sim.SHF_INSTR(warp, true, 5, 5, 0x1, 255);                                   // SHF.L.U32 R5, R5, 0x1, RZ ;
  sim.IMAD_INSTR(warp,
                 true,
                 20,
                 5,
                 12,
                 reinterpret_cast<intptr_t>(a),
                 0b001); // IMAD.WIDE.U32 R20, R5, R12, c[0x0][0x160] ;
  sim.IMAD_INSTR(warp,
                 true,
                 12,
                 7,
                 12,
                 reinterpret_cast<intptr_t>(b),
                 0b001); // IMAD.WIDE.U32 R12, R7, R12, c[0x0][0x168] ;

  sim.LDG_INSTR(warp, 128, 24, 20, 0x0);                 // LDG.E.128.SYS R24, [R20] ;
  sim.LDG_INSTR(warp, 128, 16, 12, 0x0);                 // LDG.E.128.SYS R16, [R12] ;
  sim.LDG_INSTR(warp, 128, 20, 20, 0x10);                // LDG.E.128.SYS R20, [R20+0x10] ;
  sim.LDG_INSTR(warp, 128, 12, 12, 0x10);                // LDG.E.128.SYS R12, [R12+0x10] ;
  sim.IMAD_INSTR(warp, false, 8, 255, 255, 255, 0b000);  // CS2R R8, SRZ ;
  sim.IMAD_INSTR(warp, false, 9, 255, 255, 255, 0b000);  //
  sim.IMAD_INSTR(warp, false, 10, 255, 255, 255, 0b000); // CS2R R10, SRZ ;
  sim.IMAD_INSTR(warp, false, 11, 255, 255, 255, 0b000); //
  sim.IMAD_INSTR(warp, false, 4, 255, 255, 255, 0b000);  // CS2R R4, SRZ ;
  sim.IMAD_INSTR(warp, false, 5, 255, 255, 255, 0b000);  //
  sim.IMAD_INSTR(warp, false, 6, 255, 255, 255, 0b000);  // CS2R R6, SRZ ;
  sim.IMAD_INSTR(warp, false, 7, 255, 255, 255, 0b000);  //
  sim.HMMA_INSTR_STEP0(warp,
                       8,
                       24,
                       16,
                       8);              // HMMA.884.F32.F32.STEP0 R8, R24.reuse.ROW, R16.reuse.COL, R8 ;
  sim.HMMA_INSTR_STEP1(warp,
                       10,
                       24,
                       16,
                       10);            // HMMA.884.F32.F32.STEP1 R10, R24.reuse.ROW, R16.reuse.COL, R10 ;
  sim.HMMA_INSTR_STEP2(warp,
                       4,
                       24,
                       16,
                       4);              // HMMA.884.F32.F32.STEP2 R4, R24.reuse.ROW, R16.reuse.COL, R4 ;
  sim.HMMA_INSTR_STEP3(warp, 6, 24, 16, 6);              // HMMA.884.F32.F32.STEP3 R6, R24.ROW, R16.COL, R6 ;
  sim.HMMA_INSTR_STEP0(warp,
                       8,
                       26,
                       18,
                       8);              // HMMA.884.F32.F32.STEP0 R8, R26.reuse.ROW, R18.reuse.COL, R8 ;
  sim.HMMA_INSTR_STEP1(warp,
                       10,
                       26,
                       18,
                       10);            // HMMA.884.F32.F32.STEP1 R10, R26.reuse.ROW, R18.reuse.COL, R10 ;
  sim.HMMA_INSTR_STEP2(warp,
                       4,
                       26,
                       18,
                       4);              // HMMA.884.F32.F32.STEP2 R4, R26.reuse.ROW, R18.reuse.COL, R4 ;
  sim.HMMA_INSTR_STEP3(warp, 6, 26, 18, 6);              // HMMA.884.F32.F32.STEP3 R6, R26.ROW, R18.COL, R6 ;
  sim.HMMA_INSTR_STEP0(warp,
                       8,
                       20,
                       12,
                       8);              // HMMA.884.F32.F32.STEP0 R8, R20.reuse.ROW, R12.reuse.COL, R8 ;
  sim.HMMA_INSTR_STEP1(warp,
                       10,
                       20,
                       12,
                       10);            // HMMA.884.F32.F32.STEP1 R10, R20.reuse.ROW, R12.reuse.COL, R10 ;
  sim.HMMA_INSTR_STEP2(warp,
                       4,
                       20,
                       12,
                       4);              // HMMA.884.F32.F32.STEP2 R4, R20.reuse.ROW, R12.reuse.COL, R4 ;
  sim.HMMA_INSTR_STEP3(warp, 6, 20, 12, 6);              // HMMA.884.F32.F32.STEP3 R6, R20.ROW, R12.COL, R6 ;
  sim.HMMA_INSTR_STEP0(warp,
                       8,
                       22,
                       14,
                       8);              // HMMA.884.F32.F32.STEP0 R8, R22.reuse.ROW, R14.reuse.COL, R8 ;
  sim.HMMA_INSTR_STEP1(warp,
                       10,
                       22,
                       14,
                       10);            // HMMA.884.F32.F32.STEP1 R10, R22.reuse.ROW, R14.reuse.COL, R10 ;
  sim.HMMA_INSTR_STEP2(warp,
                       4,
                       22,
                       14,
                       4);              // HMMA.884.F32.F32.STEP2 R4, R22.reuse.ROW, R14.reuse.COL, R4 ;
  sim.HMMA_INSTR_STEP3(warp, 6, 22, 14, 6);              // HMMA.884.F32.F32.STEP3 R6, R22.ROW, R14.COL, R6 ;

  sim.IMAD_INSTR(warp, false, 2, 2, 0x4, 255, 0b010);                 // IMAD.SHL.U32 R2, R2, 0x4, RZ ;
  sim.LOP3_INSTR(warp, 29, 2, 0x4, 29, 0xe2, 0b010);                  // LOP3.LUT R29, R2, 0x4, R29, 0xe2, !PT ;
  sim.LOP3_INSTR(warp, 17, 29, 0x2, 255, 0xc0, 0b010);                // LOP3.LUT R17, R29, 0x2, RZ, 0xc0, !PT ;
  sim.IMAD_INSTR(warp, false, 16, 3, 0x8, 255, 0b010);                // IMAD.SHL.U32 R16, R3, 0x8, RZ ;
  sim.LOP3_INSTR(warp, 29, 29, 0x5, 255, 0xc0, 0b010);                // LOP3.LUT R29, R29, 0x5, RZ, 0xc0, !PT ;
  sim.IMAD_INSTR(warp, false, 3, 255, 255, 255, 0b000);               // IMAD.MOV.U32 R3, RZ, RZ, RZ ;
  sim.LEA_INSTR(warp, false, false, 2, 0, 17, 255, 0b000, 0x3, 7, 7); // LEA R2, R0, R17, 0x3 ;
  sim.LOP3_INSTR(warp, 29, 16, 0x8, 29, 0xe2, 0b010);                 // LOP3.LUT R29, R16, 0x8, R29, 0xe2, !PT ;
  sim.IMAD_INSTR(warp, true, 2, 29, 0x10, 2, 0b010);                  // IMAD.WIDE.U32 R2, R29, 0x10, R2 ;
  sim.LEA_INSTR(warp, false, false, 12, 2,                            // LEA R12, P0, R2.reuse, c[0x0][0x170], 0x2 ;
                reinterpret_cast<intptr_t>(c) & 0xffffffff, 255,      //
                0b010, 0x2, 0, 7);                                    //
  sim.LEA_INSTR(warp, true, true, 13, 2,                              // LEA.HI.X R13, R2, c[0x0][0x174], R3, 0x2, P0 ;
                reinterpret_cast<intptr_t>(c) >> 32, 3,               //
                0b010, 0x2, 7, 0);                                    //
  sim.STG_INSTR(warp, 64, 12, 8, 0x0);                                // STG.E.64.SYS [R12], R8 ;
  sim.STG_INSTR(warp, 64, 12, 10, 0x80);                              // STG.E.64.SYS [R12+0x80], R10 ;
  sim.STG_INSTR(warp, 64, 12, 4, 0x10);                               // STG.E.64.SYS [R12+0x10], R4 ;
  sim.STG_INSTR(warp, 64, 12, 6, 0x90);                               // STG.E.64.SYS [R12+0x90], R6 ;
};

void deviceGEMM(GPUSimulator &sim,
                const GPUSimulator::ThreadWarp &warp,
                const f32 *a, const f32 *b, f32 *c, u32 n, u32 m, u32 k) {
  u32 front_num = warp.front();
  unit3 thread_idx;
  dim3 block_dim;
  std::tie(thread_idx, block_dim) = sim.readThreadInfo(front_num);

  // NOTE: HACK: host_gemm() set block_dim.z = WARP_SIZE
  for (const auto &thread_num : warp) {
    unit3 internal_thread_idx;
    std::tie(internal_thread_idx, std::ignore) = sim.readThreadInfo(thread_num);
    printCppError(internal_thread_idx.x != thread_idx.x ||
                      internal_thread_idx.y != thread_idx.y,
                  "set block_dim.z == WARP_SIZE should ensure all threads in a warp "
                  "have the same thread_idx.x and thread_idx.y",
                  __FILE__, __LINE__);
  }

  static const u32 N_WMMA = 16;
  u32 row = thread_idx.y * N_WMMA;
  u32 col = thread_idx.x * N_WMMA;
  u32 n_step = (k - 1) / N_WMMA + 1;

  f16 a_frag[N_WMMA][N_WMMA], b_frag[N_WMMA][N_WMMA];
  f32 c_frag[N_WMMA][N_WMMA];

  for (u32 i = 0; i < N_WMMA; i++)
    for (u32 j = 0; j < N_WMMA; j++)
      if (row + i < n && col + j < m)
        c[(row + i) * m + col + j] = 0;

  for (u32 t = 0; t < n_step; t++) {
    for (u32 i = 0; i < N_WMMA; i++)
      for (u32 j = 0; j < N_WMMA; j++) {
        a_frag[i][j] = row + i < n && t * N_WMMA + j < k
                       ? __float2half(a[(row + i) * k + t * N_WMMA + j])
                       : FP16(0);
        b_frag[i][j] = t * N_WMMA + j < k && col + i < m
                       ? __float2half(b[(t * N_WMMA + j) * m + col + i])
                       : FP16(0);
      }

    wmmaKernel(sim, warp, (f16 *) a_frag, (f16 *) b_frag, (f32 *) c_frag);

    for (u32 i = 0; i < N_WMMA; i++)
      for (u32 j = 0; j < N_WMMA; j++)
        if (row + i < n && col + j < m)
          c[(row + i) * m + col + j] += c_frag[i][j];
  }
}

// NOTE: C = A * B
void hostGEMM(const f32 *a, const f32 *b, f32 *c,
              u32 n, u32 m, u32 k, GPUSimulator &sim) {
  std::cout << "Simulating GEMM(M=" << n << ", N=" << m << ", K=" << k << ") ..." << std::endl;
  f32 *d_a, *d_b, *d_c;
  sim.cudaMalloc((void **) &d_a, n * k * sizeof(f32));
  sim.cudaMalloc((void **) &d_b, k * m * sizeof(f32));
  sim.cudaMalloc((void **) &d_c, m * n * sizeof(f32));

  sim.cudaMemcpy(d_a, (void *) a, n * k * sizeof(f32), CUDAMemcpyType::MemcpyHostToDevice);
  sim.cudaMemcpy(d_b, (void *) b, k * m * sizeof(f32), CUDAMemcpyType::MemcpyHostToDevice);

  static const u32 N_WMMA = 16;
  dim3 block_dim((m - 1) / N_WMMA + 1, (n - 1) / N_WMMA + 1, GPUSimulator::WARP_SIZE);

  sim.launchKernel(block_dim, deviceGEMM,
                   (f32 *) d_a, (f32 *) d_b, (f32 *) d_c,
                   (u32) n, (u32) m, (u32) k);

  sim.cudaMemcpy((void *) c, d_c, m * n * sizeof(f32), CUDAMemcpyType::MemcpyDeviceToHost);
  sim.cudaFree(d_a);
  sim.cudaFree(d_b);
}

}
