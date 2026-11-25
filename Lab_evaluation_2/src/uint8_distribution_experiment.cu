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

__global__ void global_u8(const std::uint8_t* in, unsigned int* hist, int n, int bins) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) atomicAdd(&hist[(static_cast<unsigned>(in[i])*bins)/256],1u);
}
__global__ void shared_u8(const std::uint8_t* in, unsigned int* hist, int n, int bins) {
    extern __shared__ unsigned int local[];
    int t=threadIdx.x, i=blockIdx.x*blockDim.x+t;
    for(int b=t;b<bins;b+=blockDim.x) local[b]=0;
    __syncthreads();
    if(i<n) atomicAdd(&local[(static_cast<unsigned>(in[i])*bins)/256],1u);
    __syncthreads();
    for(int b=t;b<bins;b+=blockDim.x) if(local[b]) atomicAdd(&hist[b],local[b]);
}
std::vector<std::uint8_t> make_data(int n, const std::string& kind, std::mt19937& rng) {
    std::vector<std::uint8_t> v(n);
    std::uniform_int_distribution<int> u(0,255);
    std::normal_distribution<double> g(128.0,30.0);
    for(int i=0;i<n;++i) {
        if(kind=="UNIFORM") v[i]=static_cast<std::uint8_t>(u(rng));
        else if(kind=="GAUSSIAN") v[i]=static_cast<std::uint8_t>(std::clamp((int)std::lround(g(rng)),0,255));
        else v[i]=(i%10==0)?static_cast<std::uint8_t>(u(rng)):128;
    }
    if(kind=="HIGHLY_SKEWED") std::shuffle(v.begin(),v.end(),rng);
    return v;
}
std::vector<unsigned int> cpu_u8(const std::vector<std::uint8_t>& v,int bins) {
    std::vector<unsigned int> h(bins,0);
    for(auto x:v) ++h[(static_cast<unsigned>(x)*bins)/256];
    return h;
}
double run_gpu(const std::vector<std::uint8_t>& v,int bins,bool shared,int repeats) {
    std::uint8_t* d=nullptr; unsigned int* dh=nullptr;
    CUDA_CHECK(cudaMalloc(&d,v.size()));
    CUDA_CHECK(cudaMalloc(&dh,bins*sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d,v.data(),v.size(),cudaMemcpyHostToDevice));
    int blocks=(static_cast<int>(v.size())+THREADS-1)/THREADS;
    cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    double total=0;
    for(int r=0;r<repeats;++r) {
        CUDA_CHECK(cudaEventRecord(s));
        CUDA_CHECK(cudaMemset(dh,0,bins*sizeof(unsigned int)));
        if(shared) shared_u8<<<blocks,THREADS,bins*sizeof(unsigned int)>>>(d,dh,(int)v.size(),bins);
        else global_u8<<<blocks,THREADS>>>(d,dh,(int)v.size(),bins);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e)); total+=ms;
    }
    std::vector<unsigned int> out(bins);
    CUDA_CHECK(cudaMemcpy(out.data(),dh,bins*sizeof(unsigned int),cudaMemcpyDeviceToHost));
    auto ref=cpu_u8(v,bins);
    if(!equal_hist(ref,out)) std::cerr<<"ERROR: GPU histogram mismatch\n";
    cudaEventDestroy(s); cudaEventDestroy(e); cudaFree(d); cudaFree(dh);
    return total/repeats;
}

int main() {
    constexpr int N=1<<24, REPEATS=10;
    std::mt19937 rng(42);
    for(const std::string kind:{"UNIFORM","GAUSSIAN","HIGHLY_SKEWED"}) {
        auto data=make_data(N,kind,rng);
        auto t0=std::chrono::steady_clock::now();
        auto ref=cpu_u8(data,256);
        auto t1=std::chrono::steady_clock::now();
        double cpu=std::chrono::duration<double,std::milli>(t1-t0).count();
        double global=run_gpu(data,256,false,REPEATS);
        double shared=run_gpu(data,256,true,REPEATS);
        std::cout<<kind<<" CPU_ms="<<cpu<<" global_ms="<<global<<" shared_ms="<<shared
                 <<" global_speedup="<<cpu/global<<" shared_speedup="<<cpu/shared
                 <<" shared/global="<<global/shared
                 <<" total="<<sum_hist(ref)<<"\n";
    }
}
