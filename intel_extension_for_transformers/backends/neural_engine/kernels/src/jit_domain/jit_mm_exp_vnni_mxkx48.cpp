//  Copyright (c) 2022 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "jit_domain/jit_mm_exp_vnni_mxkx48.hpp"

#define GET_OFF(field) offsetof(jit_mm_exp_vnni_mxkx48_t::rt_data_t<void>, field)

namespace jd {
jit_mm_exp_vnni_mxkx48_t::jit_mm_exp_vnni_mxkx48_t(const param_t& param)
    : jit_generator(),
      N_(param.dst_N),
      TW_(ceil_div(N_, 16L)),
      bias_lshift(param.bias_lshift),
      binary_add(param.binary_add),
      dt_dst(param.dt_dst),
      dsize_dst(get_data_size(dt_dst)) {
  SPARSE_LOG_IF(FATAL, N_ <= 0 || N_ > 48) << "Output dimention invalid!";
}

Xbyak::Zmm jit_mm_exp_vnni_mxkx48_t::dst_tile_Vmm(int i, int j) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS;
  const int& alloc_idx = alloc_start - i * TW_ - j;
  return Xbyak::Zmm(alloc_idx);
}
Xbyak::Zmm jit_mm_exp_vnni_mxkx48_t::TW_Vmm(int j) { return Xbyak::Zmm(j); }
Xbyak::Zmm jit_mm_exp_vnni_mxkx48_t::dst_scale_Vmm(int j) { return Xbyak::Zmm(3 + j); }

void jit_mm_exp_vnni_mxkx48_t::exp_approx_ps(const Zmm& x, const Zmm& log2e, const Xbyak::Operand& half, const Zmm& c0,
                                             const Xbyak::Operand& c1, const Xbyak::Operand& c2, const Zmm& vtemp,
                                             const Zmm& vtemp2) {
  vfmadd213ps(x, log2e, half);     // x = x * log2e + _mm512_set1_ps(0.5f);
  vcvtps2dq(vtemp, x | T_rd_sae);  //
  vcvtdq2ps(vtemp, vtemp);         // vtemp = floor(x)
  vsubps(x, x, vtemp);             // f = x1 - z;
  vmovdqu32(vtemp2, x);            //
  vfmadd213ps(vtemp2, c0, c1);     // f * c0 + c1
  vfmadd213ps(x, vtemp2, c2);      // (f * c0 + c1) * f + c2
  vscalefps(x, x, vtemp);          // f * 2^z
}

void jit_mm_exp_vnni_mxkx48_t::generate() {
  inLocalLabel();  // use local label for multiple instance
  preamble();
  mov(reg_src0, ptr[parambase + GET_OFF(src0)]);
  mov(reg_ksize.cvt32(), dword[parambase + GET_OFF(K)]);
  mov(reg_dst, ptr[parambase + GET_OFF(dst)]);
  mov(reg_ld_dst.cvt32(), dword[parambase + GET_OFF(ld_dst)]);
  mov(reg_src_b0, ptr[parambase + GET_OFF(src_b0)]);
  vpbroadcastd(vreg_log2ef, dword[rip + l_log2ef]);

  for (int j = 0; j < TW_; ++j)  // clear sum vregs
    vxorps(dst_scale_Vmm(j), dst_scale_Vmm(j), dst_scale_Vmm(j));

  xor_(reg_iterm, reg_iterm);
  L(l_mloop);
  // move/reset in bias & src1
  mov(reg_src1, ptr[parambase + GET_OFF(src1)]);
  mov(reg_tmp, ptr[parambase + GET_OFF(bias)]);
  if (bias_lshift != 0) vxorps(Xmm(vreg_temp), Xmm(vreg_temp), Xmm(vreg_temp));
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      if (i == 0) {
        if (bias_lshift != 0) {
          vpslld(dst_tile_Vmm(i, j), zword[reg_tmp + j * BYTES_ZMM], bias_lshift);
          vpsubd(dst_tile_Vmm(i, j), vreg_temp, dst_tile_Vmm(i, j));
        } else {
          vpsubd(dst_tile_Vmm(i, j), vreg_temp, zword[reg_tmp + j * BYTES_ZMM]);
        }
      } else {
        vmovdqu32(dst_tile_Vmm(i, j), dst_tile_Vmm(0, j));
      }
    }
  }

  // inner prod
  xor_(reg_iterk, reg_iterk);
  L(l_kloop);
  for (int j = 0; j < TW_; ++j) {  // move in src1
    vmovdqu8(TW_Vmm(j), ptr[reg_src1 + j * BYTES_ZMM]);
  }
  for (int i = 0; i < TH_; ++i) {
    vpbroadcastd(vreg_temp, ptr[reg_src0 + i * 4]);  // move in src0
    for (int j = 0; j < TW_; ++j) {
      vpdpbusds(dst_tile_Vmm(i, j), vreg_temp, TW_Vmm(j));
    }
  }

  lea(reg_src0, ptr[reg_src0 + TH_ * 4]);          // src0 is paded with 8
  lea(reg_src1, ptr[reg_src1 + TW_ * BYTES_ZMM]);  // src1 is paded with 48
  lea(reg_iterk, ptr[reg_iterk + 4]);
  cmp(reg_iterk, reg_ksize);  // k iteration variable
  jb(l_kloop);

  // scale & binary_add(f32) & postpo
  if (N_ % VEC != 0) {
    mov(reg_tmp.cvt16(), (1 << (N_ % VEC)) - 1);
    kmovw(mask_n, reg_tmp.cvt16());
  }
  mov(reg_tmp.cvt32(), dword[parambase + GET_OFF(scale)]);
  const auto& vreg_scale = vreg_temp;
  vpbroadcastd(vreg_scale, reg_tmp.cvt32());
  const auto& vreg_c0 = TW_Vmm(0);
  vpbroadcastd(vreg_c0, dword[rip + l_poly_c[0]]);
  vpbroadcastd(vreg_log2ef, dword[rip + l_log2ef]);
  const Zmm vreg_exp_temp[] = {TW_Vmm(1), TW_Vmm(2)};

  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      vcvtdq2ps(dst_tile_Vmm(i, j) | T_rn_sae, dst_tile_Vmm(i, j));  // s32 => f32
      if (binary_add) {                                              // f32 x scale + badd
        vfmadd213ps(dst_tile_Vmm(i, j), vreg_scale, zword_b[reg_src_b0 + i * sizeof(float)]);
      } else {
        vmulps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), vreg_scale);
      }
      exp_approx_ps(dst_tile_Vmm(i, j), vreg_log2ef, zword_b[rip + l_halff],          //
                    vreg_c0, zword_b[rip + l_poly_c[1]], zword_b[rip + l_poly_c[2]],  //
                    vreg_exp_temp[0], vreg_exp_temp[1]);
      vaddps(dst_scale_Vmm(j), dst_scale_Vmm(j), dst_tile_Vmm(i, j));

      const bool needs_mask = N_ % VEC != 0 && j == TW_ - 1;
      const Xbyak::Address out = ptr[reg_dst + j * VEC * dsize_dst] | (needs_mask ? mask_n : k0);
      switch (dt_dst) {  // move out
        case data_type::u8:
          vpmovusdb(out, dst_tile_Vmm(i, j));
          break;
        case data_type::s8:
          vpmovsdb(out, dst_tile_Vmm(i, j));
          break;
        case data_type::fp32:
          vmovups(out, dst_tile_Vmm(i, j));
          break;
        case data_type::bf16:
          vpsrld(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), 16U);
          vpmovdw(out, dst_tile_Vmm(i, j));
          break;
        default:
          SPARSE_LOG(FATAL) << "Unexpected dst type";
          break;
      }
    }
    lea(reg_dst, ptr[reg_dst + reg_ld_dst * dsize_dst]);
  }
  lea(reg_src_b0, ptr[reg_src_b0 + TH_ * sizeof(float)]);
  lea(reg_iterm, ptr[reg_iterm + TH_]);
  cmp(reg_iterm.cvt32(), dword[parambase + GET_OFF(M)]);
  jl(l_mloop);

  mov(reg_dst, ptr[parambase + GET_OFF(dst_scale)]);
  vpbroadcastd(vreg_temp, dword[rip + l_255]);
  for (int j = 0; j < TW_; ++j) {
    const bool needs_mask = N_ % VEC != 0 && j == TW_ - 1;
    const Xbyak::Address out = ptr[reg_dst + j * VEC * sizeof(float)] | (needs_mask ? mask_n : k0);
    vdivps(dst_scale_Vmm(j), vreg_temp, dst_scale_Vmm(j));
    vmovups(out, dst_scale_Vmm(j));
  }

  postamble();  // ret

  const float constants[] = {1.442695f, .5f, 0.240226507f, 0.452920674f, 0.713483036f, 255.f};
  L(l_log2ef);
  db(reinterpret_cast<const uint8_t*>(&constants[0]), sizeof(float));
  L(l_halff);
  db(reinterpret_cast<const uint8_t*>(&constants[1]), sizeof(float));
  L(l_poly_c[0]);
  db(reinterpret_cast<const uint8_t*>(&constants[2]), sizeof(float));
  L(l_poly_c[1]);
  db(reinterpret_cast<const uint8_t*>(&constants[3]), sizeof(float));
  L(l_poly_c[2]);
  db(reinterpret_cast<const uint8_t*>(&constants[4]), sizeof(float));
  L(l_255);
  db(reinterpret_cast<const uint8_t*>(&constants[5]), sizeof(float));

  outLocalLabel();  // end of local label
}
}  // namespace jd
