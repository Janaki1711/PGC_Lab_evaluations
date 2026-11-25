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

__global__ void global_float_hist(const float* in, unsigned int* hist, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) atomicAdd(&hist[float_bin(in[i])],1u);
}
int main() {
    constexpr int N=1<<20, BLOCKS=(N+THREADS-1)/THREADS, REPEATS=20;
    std::vector<float> h(N);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-4.f,4.f);
    for(auto& x:h) x=dist(rng);
    std::vector<unsigned int> ref(BINS,0);
    for(float x:h) ++ref[float_bin(x)];
    float *d=nullptr; unsigned int *dh=nullptr;
    CUDA_CHECK(cudaMalloc(&d,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dh,BINS*sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d,h.data(),N*sizeof(float),cudaMemcpyHostToDevice));
    cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    double total=0;
    for(int r=0;r<REPEATS;++r){
        CUDA_CHECK(cudaEventRecord(s)); CUDA_CHECK(cudaMemset(dh,0,BINS*sizeof(unsigned int)));
        global_float_hist<<<BLOCKS,THREADS>>>(d,dh,N);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e)); total+=ms;
    }
    std::vector<unsigned int> out(BINS);
    CUDA_CHECK(cudaMemcpy(out.data(),dh,BINS*sizeof(unsigned int),cudaMemcpyDeviceToHost));
    bool ok=equal_hist(ref,out);
    std::cout<<std::fixed<<std::setprecision(4)<<"N="<<N<<" bins="<<BINS
             <<" average GPU ms="<<total/REPEATS<<" total="<<sum_hist(out)
             <<" correctness="<<(ok?"PASSED":"FAILED")<<"\n";
    cudaEventDestroy(s); cudaEventDestroy(e); cudaFree(d); cudaFree(dh);
    return ok?0:1;
}
