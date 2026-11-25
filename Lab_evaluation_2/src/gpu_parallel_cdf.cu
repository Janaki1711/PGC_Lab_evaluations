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

__device__ __constant__ float DEVICE_PERCENTILES[3]={0.99f,0.999f,0.9999f};
__global__ void hist_kernel(const float*x,unsigned int*h,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int b=min(255,(int)(fabsf(x[i])*64.f));atomicAdd(&h[b],1u);}}
__global__ void scan_kernel(const unsigned int*in,unsigned long long*out){__shared__ unsigned long long s[256];int t=threadIdx.x;s[t]=in[t];__syncthreads();for(int d=1;d<256;d<<=1){unsigned long long v=t>=d?s[t-d]:0;__syncthreads();s[t]+=v;__syncthreads();}out[t]=s[t];}
__global__ void percentile_kernel(const unsigned long long*cdf,unsigned long long n,int*out){int p=threadIdx.x;if(p<3){unsigned long long target=(unsigned long long)(DEVICE_PERCENTILES[p]*n);int b=0;while(b<255&&cdf[b]<target)++b;out[p]=b;}}
int main(){const int N=1<<20;std::vector<float>x(N);std::mt19937 r(42);std::normal_distribution<float>g(0,1);for(auto&v:x)v=g(r);float*d;unsigned int*dh;unsigned long long*dc;int*dp;CUDA_CHECK(cudaMalloc(&d,N*sizeof(float)));CUDA_CHECK(cudaMalloc(&dh,256*sizeof(unsigned int)));CUDA_CHECK(cudaMalloc(&dc,256*sizeof(unsigned long long)));CUDA_CHECK(cudaMalloc(&dp,3*sizeof(int)));CUDA_CHECK(cudaMemcpy(d,x.data(),N*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemset(dh,0,256*sizeof(unsigned int)));hist_kernel<<<(N+255)/256,256>>>(d,dh,N);scan_kernel<<<1,256>>>(dh,dc);percentile_kernel<<<1,3>>>(dc,N,dp);CUDA_CHECK(cudaGetLastError());std::array<int,3>p{};CUDA_CHECK(cudaMemcpy(p.data(),dp,3*sizeof(int),cudaMemcpyDeviceToHost));std::cout<<"GPU percentile bins: "<<p[0]<<", "<<p[1]<<", "<<p[2]<<"\n";cudaFree(d);cudaFree(dh);cudaFree(dc);cudaFree(dp);}
