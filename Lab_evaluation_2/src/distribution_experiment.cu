#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
 std::cerr << "CUDA error " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
 std::exit(2); }} while(0)

constexpr int BINS = 256;
constexpr int THREADS = 256;

__host__ __device__ inline int float_bin(float x, float lo=-4.f, float hi=4.f) {
    int b = static_cast<int>((x-lo) * (BINS/(hi-lo)));
    return b < 0 ? 0 : (b >= BINS ? BINS-1 : b);
}

template<class T>
bool equal_hist(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin());
}

template<class T>
std::uint64_t sum_hist(const std::vector<T>& h) {
    return std::accumulate(h.begin(),h.end(),std::uint64_t{0});
}

template<class T>
double elapsed_ms(T fn, int repeats) {
    double total=0;
    for(int i=0;i<repeats;++i) total += fn();
    return total/repeats;
}

__global__ void gh(const float* x,unsigned int* h,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&h[float_bin(x[i])],1u);}
__global__ void sh(const float* x,unsigned int* h,int n){__shared__ unsigned int l[BINS];int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;l[t]=0;__syncthreads();if(i<n)atomicAdd(&l[float_bin(x[i])],1u);__syncthreads();atomicAdd(&h[t],l[t]);}
int main(){
 const int N=1<<20, blocks=(N+THREADS-1)/THREADS; std::mt19937 r(42);
 std::normal_distribution<float> normal(0,1); std::uniform_real_distribution<float> uni(-4,4); std::exponential_distribution<float> expd(1.5f);
 for(std::string kind:{"UNIFORM","GAUSSIAN","RELU_LIKE","SKEWED_EXP"}){
  std::vector<float>x(N);for(auto&v:x){if(kind=="UNIFORM")v=uni(r);else if(kind=="GAUSSIAN")v=std::clamp(normal(r),-4.f,4.f);else if(kind=="RELU_LIKE")v=std::min(4.f,std::max(0.f,normal(r)));else v=std::clamp(-4.f+expd(r),-4.f,4.f);}
  std::vector<unsigned int> ref(BINS);for(float v:x)++ref[float_bin(v)];
  float*d;unsigned int*dh;CUDA_CHECK(cudaMalloc(&d,N*sizeof(float)));CUDA_CHECK(cudaMalloc(&dh,BINS*sizeof(unsigned int)));CUDA_CHECK(cudaMemcpy(d,x.data(),N*sizeof(float),cudaMemcpyHostToDevice));
  for(int mode=0;mode<2;++mode){cudaEvent_t s,e;CUDA_CHECK(cudaEventCreate(&s));CUDA_CHECK(cudaEventCreate(&e));double total=0;int reps=30;
   for(int q=0;q<reps;++q){CUDA_CHECK(cudaEventRecord(s));CUDA_CHECK(cudaMemset(dh,0,BINS*sizeof(unsigned int)));if(mode==0)gh<<<blocks,THREADS>>>(d,dh,N);else sh<<<blocks,THREADS>>>(d,dh,N);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaEventRecord(e));CUDA_CHECK(cudaEventSynchronize(e));float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));total+=ms;}
   std::vector<unsigned int>out(BINS);CUDA_CHECK(cudaMemcpy(out.data(),dh,BINS*sizeof(unsigned int),cudaMemcpyDeviceToHost));std::cout<<kind<<" "<<(mode?"SHARED":"GLOBAL")<<" ms="<<total/reps<<" correct="<<equal_hist(ref,out)<<"\n";cudaEventDestroy(s);cudaEventDestroy(e);
  }cudaFree(d);cudaFree(dh);
 }
}
