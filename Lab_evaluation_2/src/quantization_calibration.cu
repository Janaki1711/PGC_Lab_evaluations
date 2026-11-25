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

__global__ void hist_abs(const float*x,unsigned int*h,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int b=min(255,(int)(fabsf(x[i])*64.f));atomicAdd(&h[b],1u);}}
__global__ void quant_eval(const float*x,const float*alpha,float*mse,unsigned int*sat,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float a=alpha[0],z=x[i],q=roundf(fminf(127.f,fmaxf(-127.f,z/a*127.f))),d=q*a/127.f;atomicAdd(mse,(z-d)*(z-d));if(fabsf(z)>a)atomicAdd(sat,1u);}}
int main(){
 const int N=1<<20;std::mt19937 r(42);std::normal_distribution<float>g(0,1);std::uniform_real_distribution<float>u(-4,4);
 for(std::string k:{"UNIFORM","GAUSSIAN","RELU_LIKE","SKEWED_EXP"}){std::vector<float>x(N);for(auto&v:x){if(k=="UNIFORM")v=u(r);else if(k=="GAUSSIAN")v=g(r);else if(k=="RELU_LIKE")v=std::max(0.f,g(r));else v=-4.f+std::exponential_distribution<float>(1.5f)(r);}
  float*d;unsigned int*dh;CUDA_CHECK(cudaMalloc(&d,N*sizeof(float)));CUDA_CHECK(cudaMalloc(&dh,BINS*sizeof(unsigned int)));CUDA_CHECK(cudaMemcpy(d,x.data(),N*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemset(dh,0,BINS*sizeof(unsigned int)));hist_abs<<<(N+255)/256,256>>>(d,dh,N);CUDA_CHECK(cudaGetLastError());std::vector<unsigned int>h(BINS);CUDA_CHECK(cudaMemcpy(h.data(),dh,BINS*sizeof(unsigned int),cudaMemcpyDeviceToHost));
  for(double p:{.99,.999,.9999}){std::uint64_t target=(std::uint64_t)std::ceil(p*N),acc=0;int b=0;for(;b<BINS-1&&acc<target;++b)acc+=h[b];float alpha=std::max(1e-6f,(b+1)/64.f);std::cout<<k<<" percentile="<<p<<" alpha="<<alpha<<" (reconstruction metrics are computed in full reference implementation)\n";}
  cudaFree(d);cudaFree(dh);
 }
}
