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

__global__ void hist_global(const unsigned char*x,unsigned int*h,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&h[x[i]],1u);}
__global__ void hist_shared(const unsigned char*x,unsigned int*h,int n){__shared__ unsigned int l[256];int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;l[t]=0;__syncthreads();if(i<n)atomicAdd(&l[x[i]],1u);__syncthreads();atomicAdd(&h[t],l[t]);}
int main(int argc,char**argv){
 if(argc<2){std::cerr<<"Usage: div2k_histogram_benchmark <div2k_rgb_u8.bin>\n";return 1;}
 std::ifstream f(argv[1],std::ios::binary|std::ios::ate);if(!f){std::cerr<<"Cannot open input\n";return 1;}auto bytes=(std::uint64_t)f.tellg();f.seekg(0);
 constexpr std::size_t CHUNK=64ull*1024*1024;std::vector<unsigned char>buf(CHUNK);std::array<std::uint64_t,256>cpu{},gpuG{},gpuS{};unsigned char*d;unsigned int*dh;CUDA_CHECK(cudaMalloc(&d,CHUNK));CUDA_CHECK(cudaMalloc(&dh,256*sizeof(unsigned int)));
 auto wall0=std::chrono::steady_clock::now();std::uint64_t done=0;
 while(done<bytes){std::size_t n=(std::size_t)std::min<std::uint64_t>(CHUNK,bytes-done);f.read((char*)buf.data(),n);if((std::size_t)f.gcount()!=n){std::cerr<<"Short read\n";return 2;}for(std::size_t j=0;j<n;++j)++cpu[buf[j]];CUDA_CHECK(cudaMemcpy(d,buf.data(),n,cudaMemcpyHostToDevice));int blocks=(int)((n+255)/256);
  for(int mode=0;mode<2;++mode){CUDA_CHECK(cudaMemset(dh,0,256*sizeof(unsigned int)));if(mode==0)hist_global<<<blocks,256>>>(d,dh,(int)n);else hist_shared<<<blocks,256>>>(d,dh,(int)n);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());std::array<unsigned int,256>part{};CUDA_CHECK(cudaMemcpy(part.data(),dh,256*sizeof(unsigned int),cudaMemcpyDeviceToHost));auto&dst=mode?gpuS:gpuG;for(int b=0;b<256;++b)dst[b]+=part[b];}
  done+=n;std::cerr<<"\rProcessed "<<done<<" / "<<bytes<<" bytes"<<std::flush;
 }
 auto wall1=std::chrono::steady_clock::now();auto ms=std::chrono::duration<double,std::milli>(wall1-wall0).count();bool ok=true;for(int b=0;b<256;++b)ok&=(cpu[b]==gpuG[b]&&cpu[b]==gpuS[b]);std::cout<<"\nBytes="<<bytes<<" wall_ms="<<ms<<" correctness="<<(ok?"PASSED":"FAILED")<<"\n";cudaFree(d);cudaFree(dh);return ok?0:1;
}
