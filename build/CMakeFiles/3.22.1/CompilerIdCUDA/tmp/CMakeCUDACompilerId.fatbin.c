#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x00000000000003a8,0x0000004001010002,0x0000000000000368\n"
".quad 0x0000000000000000,0x0000004b00010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007300be0002,0x0000000000000000,0x0000000000000000,0x00000000000001e8\n"
".quad 0x00000040004b054b,0x0001000600400000,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x67756265642e006f,0x2e00656d6172665f,0x612e6c65722e766e,0x2e00006e6f697463\n"
".quad 0x6261747274736873,0x6261747274732e00,0x6261746d79732e00,0x6261746d79732e00\n"
".quad 0x2e0078646e68735f,0x006f666e692e766e,0x665f67756265642e,0x766e2e00656d6172\n"
".quad 0x7463612e6c65722e,0x00000000006e6f69,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x000500030000003f,0x0000000000000000,0x0000000000000000\n"
".quad 0x000000000000004b,0x222f0a1008020200,0x0000000008000000,0x0000000008080000\n"
".quad 0x0000000008100000,0x0000000008180000,0x0000000008200000,0x0000000008280000\n"
".quad 0x0000000008300000,0x0000000008380000,0x0000000008000001,0x0000000008080001\n"
".quad 0x0000000008100001,0x0000000008180001,0x0000000008200001,0x0000000008280001\n"
".quad 0x0000000008300001,0x0000000008380001,0x0000000008000002,0x0000000008080002\n"
".quad 0x0000000008100002,0x0000000008180002,0x0000000008200002,0x0000000008280002\n"
".quad 0x0000000008300002,0x0000000008380002,0x0000002c14000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000300000001\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000040,0x000000000000004e\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x000000030000000b\n"
".quad 0x0000000000000000,0x0000000000000000,0x000000000000008e,0x000000000000004e\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x0000000200000013\n"
".quad 0x0000000000000000,0x0000000000000000,0x00000000000000e0,0x0000000000000030\n"
".quad 0x0000000200000002,0x0000000000000008,0x0000000000000018,0x0000000100000032\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000110,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x7000000b0000003f\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000110,0x00000000000000d8\n"
".quad 0x0000000000000000, 0x0000000000000008, 0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[119];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif