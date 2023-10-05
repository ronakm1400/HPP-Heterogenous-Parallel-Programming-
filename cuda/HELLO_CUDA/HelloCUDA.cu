//Standard headers
#include<stdio.h>

//For Cuda Headers
#include<cuda.h>

//Global Variable
const int iNumberOfArrayElements = 5;

//For CPU RAM
float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;

//For GPU RAM
float* deviceInput1 = NULL;
float* deviceInput2 = NULL;
float* deviceOutput = NULL;

//CUDA kernel
__global__ void vecAddGPU(float* in1,float* in2,float* out,int len)
{
	int i = 0;
	
	i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < len)
	{
		out[i] = in1[i] + in2[i];
	}

}

void cleanup(void)
{
	if(deviceOutput)
	{
		cudaFree(deviceOutput);
		deviceOutput = NULL;
	}
	
	if(deviceInput2)
	{
		cudaFree(deviceInput2);
		deviceInput2 = NULL;
	}
	
	if(deviceInput1)
	{
		cudaFree(deviceInput1);
		deviceInput1 = NULL;	
	}
	
	if(hostOutput)
	{
		free(hostOutput);
		hostOutput = NULL;
	}
	
	if(hostInput2)
	{
		free(hostInput2);
		hostInput2 = NULL;
	}
	
	if(hostInput1)
	{
		free(hostInput1);
		hostInput1 = NULL;
	}
}

int main(void)
{
	
	printf("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");
	printf("\t\tFirst CUDA Application\n");
	printf("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");
	
	
	void cleanup(void);		
	
	int iSize = 0;
	int iCnt = 0;
	
	iSize = iNumberOfArrayElements * sizeof(float);
	cudaError_t result = cudaSuccess;
	
	hostInput1 = (float*)malloc(iSize);
	if(hostInput1 == NULL)
	{
		printf("Host memory allocation is failed for hostInput1 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	hostInput2 = (float*)malloc(iSize);
	if(hostInput2 == NULL)
	{
		printf("Host memory allocation is failed for hostInput2 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	hostOutput = (float*)malloc(iSize);
	if(hostOutput == NULL)
	{
		printf("Host memory allocation is failed for hostOutput array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	hostInput1[0] = 101.0;
	hostInput1[1] = 102.0;
	hostInput1[2] = 103.0;
	hostInput1[3] = 104.0;
	hostInput1[4] = 105.0;
	
	hostInput2[0] = 201.0;
	hostInput2[1] = 202.0;
	hostInput2[2] = 203.0;
	hostInput2[3] = 204.0;
	hostInput2[4] = 205.0;
	
	result = cudaMalloc((void **)&deviceInput1,iSize);
	if(result != cudaSuccess)
	{
		printf("Device memory allocation is failed to allocate memory due to %s in file %s at line %d.\n",cudaGetErrorString(result),__FILE__,__LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}	
	
	result = cudaMalloc((void **)&deviceInput2,iSize);
	if(result != cudaSuccess)
	{
		printf("Device memory allocation is failed for deviceInput2 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}	
	
	result = cudaMalloc((void **)&deviceOutput,iSize);
	if(result != cudaSuccess)
	{
		printf("Device memory allocation is failed for deviceInput2 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}	
	 	  	
	result = cudaMemcpy(deviceInput1,hostInput1,iSize,cudaMemcpyHostToDevice);
	if(result != cudaSuccess)
	{
		printf("Device memory allocation is failed for deviceInput1 array.\n");
		cleanup();
		exit(EXIT_FAILURE);			
	}
	
	result = cudaMemcpy(deviceInput2,hostInput2,iSize,cudaMemcpyHostToDevice);
	if(result != cudaSuccess)
	{
		printf("Device memory allocation is failed for deviceInput2 array.\n");
		cleanup();
		exit(EXIT_FAILURE);			
	}
	
	dim3 dimGrid = dim3(iNumberOfArrayElements,1,1);
	dim3 dimBlock = dim3(1,1,1);
	
	//CUDA Kernel for vector addition
	//Kernel execution configuration
	vecAddGPU <<<dimGrid,dimBlock>>> (deviceInput1,deviceInput2,deviceOutput,iNumberOfArrayElements);
	
	//Copy data from device array into host array
	result = cudaMemcpy(hostOutput,deviceOutput,iSize,cudaMemcpyDeviceToHost);
	if(result != cudaSuccess)
	{
		printf("Device to host data copy is failed for %s reason in %s file at %d \n",cudaGetErrorString(result),__FILE__,__LINE__);
		cleanup();
		exit(EXIT_FAILURE);	
	}
	
	//Vector addition on host
	printf("\nVector Addition Performed On Host : \n\n"); 
	for(iCnt = 0; iCnt < iNumberOfArrayElements; iCnt++)
	{
		printf("%f + %f = %f\n",hostInput1[iCnt],hostInput2[iCnt],hostOutput[iCnt]);
	}	
	
	cleanup();
		
	exit(EXIT_SUCCESS);
}
