/* Diagnostic cross-correlation, with explicit operation ordering.
 * This is not a deployed kernel. No native library code is copied here.
 */
#include <math.h>
#include <stddef.h>

void course_conv_forward(const float *x, const float *w, const float *bias,
                        float *out, int batch, int channels, int height, int width,
                        int outputs, int kh, int kw, int stride, int padding,
                        int bias_first, int fused) {
    const int oh = (height + 2 * padding - kh) / stride + 1;
    const int ow = (width + 2 * padding - kw) / stride + 1;
    for (int n=0;n<batch;n++) for (int o=0;o<outputs;o++)
        for (int y=0;y<oh;y++) for (int z=0;z<ow;z++) {
            float sum = bias_first ? bias[o] : 0.0f;
            for (int c=0;c<channels;c++) for (int ky=0;ky<kh;ky++)
                for (int kx=0;kx<kw;kx++) {
                    int iy=y*stride+ky-padding, ix=z*stride+kx-padding;
                    float a = iy>=0 && iy<height && ix>=0 && ix<width
                        ? x[((n*channels+c)*height+iy)*width+ix] : 0.0f;
                    float b = w[((o*channels+c)*kh+ky)*kw+kx];
                    sum = fused ? fmaf(a,b,sum) : sum+a*b;
                }
            if (!bias_first) sum += bias[o];
            out[((n*outputs+o)*oh+y)*ow+z] = sum;
        }
}
