/* Diagnostic port of NNPACK's ARM Fourier convolution arithmetic.
 * The upstream transforms are compiled unchanged from the pinned source.
 * This narrow kernel is not part of the deployed adapter.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

void nnp_fft16x16_with_offset__psimd(const float*, float*, size_t, size_t,
    uint32_t, uint32_t, uint32_t, uint32_t);
void nnp_ifft16x16_with_bias__psimd(const float*, float*, const float*,
    size_t, size_t, uint32_t, uint32_t);

int course_nnpack_fft16(const float *input, const float *weight,
    const float *bias, float *output, int n, int ic, int ih, int iw, int oc) {
  /* Original small MNIST preset: 5x5, unit stride, no padding. */
  const int kh = 5, kw = 5, oh = ih - kh + 1, ow = iw - kw + 1;
  if (n < 1 || ic < 1 || oc < 1 || oh < 1 || ow < 1) return 1;
  float *kernels = malloc((size_t)oc * ic * 256 * sizeof(float));
  float *inputs = malloc((size_t)ic * 256 * sizeof(float));
  if (!kernels || !inputs) { free(kernels); free(inputs); return 2; }
  for (int o = 0; o < oc; o++) for (int c = 0; c < ic; c++) {
    nnp_fft16x16_with_offset__psimd(weight + (o * ic + c) * kh * kw,
      kernels + (o * ic + c) * 256, kw, 8 * sizeof(float), kh, kw, 0, 0);
  }
  for (int sample = 0; sample < n; sample++) {
    for (int y = 0; y < oh; y += 12) for (int x = 0; x < ow; x += 12) {
      for (int c = 0; c < ic; c++) {
        nnp_fft16x16_with_offset__psimd(input + ((sample * ic + c) * ih + y) * iw + x,
          inputs + c * 256, iw, 8 * sizeof(float),
          ih - y < 16 ? ih - y : 16, iw - x < 16 ? iw - x : 16, 0, 0);
      }
      for (int o = 0; o < oc; o++) {
        float product[256] __attribute__((aligned(64))) = {0};
        for (int tuple = 0; tuple < 32; tuple++) for (int c = 0; c < ic; c++) {
          const float *a = inputs + c * 256 + tuple * 8;
          const float *b = kernels + (o * ic + c) * 256 + tuple * 8;
          float *result = product + tuple * 8;
          for (int lane = 0; lane < 4; lane++) {
            result[lane] = fmaf(a[lane], b[lane], result[lane]);
            result[lane + 4] = fmaf(a[lane + 4],
              tuple == 0 && lane < 2 ? b[lane + 4] : b[lane], result[lane + 4]);
            if (tuple != 0 || lane >= 2) {
              result[lane] = fmaf(a[lane + 4], b[lane + 4], result[lane]);
              result[lane + 4] = fmaf(-a[lane], b[lane + 4], result[lane + 4]);
            }
          }
        }
        nnp_ifft16x16_with_bias__psimd(product, output + ((sample * oc + o) * oh + y) * ow + x,
          bias + o, 8 * sizeof(float), ow, oh - y < 12 ? oh - y : 12, ow - x < 12 ? ow - x : 12);
      }
    }
  }
  free(kernels); free(inputs); return 0;
}
