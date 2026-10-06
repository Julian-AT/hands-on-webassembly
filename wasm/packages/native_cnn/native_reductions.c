/* Independent scalar spelling of the pinned Torch SumKernel.cpp reductions.
 * Chunking and combination order are retained; this diagnostic is not deployed.
 */
#include <stddef.h>
#include <math.h>

static float cascade(const float *data,int stride,int count){
 int log=0;for(int value=count-1;value>0;value>>=1)log++;
 int power=log/4;if(power<4)power=4;
 int step=1<<power,mask=step-1,i=0;
 float acc[4]={0};
 while(i+step<=count){
  for(int j=0;j<step;j++,i++)acc[0]+=data[i*stride];
  for(int level=1;level<4;level++){
   acc[level]+=acc[level-1];acc[level-1]=0;
   if(i&(mask<<(level*power)))break;
  }
 }
 while(i<count){acc[0]+=data[i*stride];i++;}
 for(int level=1;level<4;level++)acc[0]+=acc[level];
 return acc[0];
}

void course_bias_outer(const float *g,float *out,int rows,int columns){
 /* Complete blocks of four native four-float vectors. Remaining columns use
  * the already-verified scalar row_sum implementation in native_linear. */
 for(int col=0;col<(columns/16)*16;col++)out[col]=cascade(g+col,columns,rows);
}

static float inner_sum(const float *g,int size){
 int vectors=size/4,groups=vectors/4;
 float partial[4];
 for(int lane=0;lane<4;lane++){
  float sum=cascade(g+lane,16,groups);
  for(int stream=1;stream<4;stream++)sum+=cascade(g+stream*4+lane,16,groups);
  partial[lane]=sum;
 }
 float result=0;
 for(int i=vectors*4;i<size;i++)result+=g[i];
 for(int lane=0;lane<4;lane++)result+=partial[lane];
 return result;
}

void course_conv_bias(const float *g,float *out,int batch,int channels,int height,int width){
 const int spatial=height*width;
 /* This candidate requires the original 24x24 shape (144 complete vectors). */
 for(int c=0;c<channels;c++){
  float sum=0;
  for(int n=0;n<batch;n++)sum+=inner_sum(g+(n*channels+c)*spatial,spatial);
  out[c]=sum;
 }
}

void course_conv_weight(const float *x,const float *g,float *out,
 int batch,int channels,int height,int width,int outputs){
 const int oh=height-4,ow=width-4;
 for(int i=0;i<outputs*channels*25;i++)out[i]=0;
 for(int n=0;n<batch;n++)for(int o=0;o<outputs;o++)
  for(int y=0;y<oh;y++)for(int z=0;z<ow;z++){
   float incoming=g[((n*outputs+o)*oh+y)*ow+z];
   if(incoming==0)continue;
   for(int c=0;c<channels;c++)for(int ky=0;ky<5;ky++)for(int kx=0;kx<5;kx++){
    int wi=((o*channels+c)*5+ky)*5+kx;
    out[wi]=fmaf(incoming,x[((n*channels+c)*height+y+ky)*width+z+kx],out[wi]);
   }
  }
}
