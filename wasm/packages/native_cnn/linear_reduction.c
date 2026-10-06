/* Isolated arithmetic-order diagnostic; no library code is copied. */
#include <math.h>
void course_linear_order(const float *x,const float *w,const float *b,float *out,
 int rows,int cols,int k,int lanes,int bias_first,int tree,int contiguous){
 for(int r=0;r<rows;r++)for(int c=0;c<cols;c++){
  float acc[32]={0};if(bias_first)acc[0]=b[c];
  for(int j=0;j<k;j++){
   int lane=contiguous ? j/((k+lanes-1)/lanes) : j%lanes;
   acc[lane]=fmaf(x[r*k+j],w[c*k+j],acc[lane]);
  }
  if(tree==1)for(int step=1;step<lanes;step*=2)for(int j=0;j+step<lanes;j+=step*2)acc[j]+=acc[j+step];
  else if(tree==2)for(int half=lanes/2;half;half/=2)for(int j=0;j<half;j++)acc[j]+=acc[j+half];
  else for(int j=1;j<lanes;j++)acc[0]+=acc[j];
  out[r*cols+c]=bias_first?acc[0]:acc[0]+b[c];
 }
}
/* The observed short-K native kernel uses two streams for complete eight-row
 * output blocks, with a serial remainder. Dispatch boundaries remain diagnostic.
 */
void course_linear_split(const float *x,const float *w,const float *b,float *out,
 int rows,int cols,int k){
 for(int r=0;r<rows;r++)for(int c=0;c<cols;c++){
  float even=b[c],odd=0.0f;
  int split=k<=64 && c<(cols/8)*8;
  for(int j=0;j<k;j++){
   if(split && j%2)odd=fmaf(x[r*k+j],w[c*k+j],odd);
   else even=fmaf(x[r*k+j],w[c*k+j],even);
  }
  out[r*cols+c]=split?even+odd:even;
 }
}
