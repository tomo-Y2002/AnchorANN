#include "textflag.h"

// Four independent centroids per SIMD register. Accumulation along dimensions
// is in the original order (no FMA or reassociation of each distance).
TEXT ·distancesAVX2(SB), NOSPLIT, $0-96
 MOVQ points_base+0(FP), R8
 MOVQ centers_base+24(FP), R9
 MOVQ costs_base+48(FP), R10
 MOVQ rows+72(FP), R11
 MOVQ count+80(FP), R12
 MOVQ dim+88(FP), R13
 XORQ AX, AX
 group:
 CMPQ AX, R12
 JGE done
 MOVQ R9, DX
 XORQ BX, BX
 row:
 CMPQ BX, R11
 JGE nextgroup
 MOVQ BX, SI
 IMULQ R13, SI
 LEAQ (R8)(SI*8), SI
 MOVQ DX, DI
 VXORPD Y0, Y0, Y0
 MOVQ R13, CX
 dimension:
 VBROADCASTSD (SI), Y1
 VMOVUPD (DI), Y2
 VSUBPD Y2, Y1, Y1
 VMULPD Y1, Y1, Y1
 VADDPD Y1, Y0, Y0
 ADDQ $8, SI
 ADDQ $32, DI
 DECQ CX
 JNZ dimension
 MOVQ BX, SI
 IMULQ R12, SI
 ADDQ AX, SI
 VMOVUPD Y0, (R10)(SI*8)
 INCQ BX
 JMP row
 nextgroup:
 MOVQ R13, SI
 SHLQ $5, SI
 ADDQ SI, R9
 ADDQ $4, AX
 JMP group
 done:
 VZEROUPPER
 RET
