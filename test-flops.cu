#include <stdio.h>
#include <cuda_runtime.h>

// A utility function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// A utility function to get the device properties
void getDeviceProperties(int device, cudaDeviceProp* prop) {
    cudaError_t err = cudaGetDeviceProperties(prop, device);
    checkCudaError(err, "cudaGetDeviceProperties");
}

// A utility function to get the SM count of the device
int getSMCount(int device) {
    cudaDeviceProp prop;
    getDeviceProperties(device, &prop);
    return prop.multiProcessorCount;
}

// A utility function to get the core count per SM of the device
int getCoreCountPerSM(int device) {
    cudaDeviceProp prop;
    getDeviceProperties(device, &prop);
    int cores = 0;
    switch (prop.major) {
        case 3: // Kepler
            cores = 192;
            break;
        case 5: // Maxwell
            cores = 128;
            break;
        case 6: // Pascal
            if (prop.minor == 0) cores = 64; 
            else if (prop.minor == 1) cores = 128;
            else if (prop.minor == 2) cores = 128;
            break;
        case 7: // Volta
            if (prop.minor == 0) cores = 64;
            else if (prop.minor == 5) cores = 64;
            break;
        case 8: // Ampere
            if (prop.minor == 0) cores = 64;
            else if (prop.minor == 6) cores = 128;
            break;
        default:
            fprintf(stderr, "Unknown device architecture: %d.%d\n", prop.major, prop.minor);
            exit(EXIT_FAILURE);
    }
    return cores;
}

// A utility function to get the core frequency in KHz of the device
int getCoreFrequency(int device) {
    cudaDeviceProp prop;
    getDeviceProperties(device, &prop);
    return prop.clockRate;
}

// A utility function to get the peak FLOPS of the device
// This is based on the core count, core frequency, tensor core count, tensor core frequency, and tensor core throughput
double getPeakFlops(int device, cudaDataType_t type) {
    int sm_count = getSMCount(device); // get the number of SMs
    int core_count_per_sm = getCoreCountPerSM(device); // get the number of cores per SM
    int core_frequency = getCoreFrequency(device); // get the core frequency in MHz
    double peak_flops = 0.0; // initialize the peak FLOPS
    // calculate the peak FLOPS based on the data type
    switch (type) {
        case CUDA_R_16F: // FP16
            peak_flops = (double)sm_count * core_count_per_sm * core_frequency * 1e3 * 4;
            break;
        case CUDA_R_32F: // FP32
            peak_flops = (double)sm_count * core_count_per_sm * core_frequency * 1e3 * 2;
            break;
        default:
            fprintf(stderr, "Unsupported data type for peak FLOPS: %d\n", type);
            exit(EXIT_FAILURE);
    }
    return peak_flops;
}

// A main function to test the peak FLOPS of the device
int main() {
    int device = 0; // use the first device by default
    cudaDataType_t type = CUDA_R_32F; // use FP16 data type by default
    
    // get the device name
    cudaDeviceProp prop;
    getDeviceProperties(device, &prop);
    printf("Device name: %s\n", prop.name);
    // get the peak FLOPS
    double peak_flops = getPeakFlops(device, type);
    printf("Peak FLOPS: %.2f GFLOPS\n", peak_flops / 1e9);
    return 0;
}
