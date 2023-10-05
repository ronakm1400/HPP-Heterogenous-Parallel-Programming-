#include<stdio.h>	
#include<cuda.h>

//Header file to calculate time required to perform operations on CPU and GPU
#include "helper_timer.h"

const int iNumberOfArrayElements = 11444777;

float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;
float* gold = NULL;

float* deviceInput1 = NULL;
float* deviceInput2 = NULL;
float* deviceOutput = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

__global__ void vecAddGPU(float* in1,float* in2,float* out,int len)
{
	int i = 0;
	
	i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < len)
	{
		out[i] = in1[i] + in2[i];
	}

}

void fillFloatArrayWithRandomNumbers(float* arr,int len)
{
	//code
	int iCnt = 0;
	
	//RAND_MAX = 32764
	
	const float fScale = 1.0f / (float)RAND_MAX;
	for(iCnt = 0; iCnt < len; iCnt++)
	{
		arr[iCnt] = fScale * rand();
	}
	
}

void vecAddCPU(const float* arr1,const float* arr2,float* out,int len)
{
	int iCnt = 0;
	
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	
	for(iCnt = 0; iCnt < len; iCnt++)
	{
		out[iCnt] = arr1[iCnt] + arr2[iCnt];
	}
	
	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;
	
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
	
	if(gold)
	{
		free(gold);
		gold = NULL;
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
	printf("\t\tVector addition on CPU & GPU\n");
	printf("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");

	// Function declarations
	
	void fillFloatArrayWithRandomNumbers(float*,int);
	void vecAddCPU(const float*,const float*,float*,int);
	void cleanup(void);
	
	//Variable declarations
	
	int iSize = 0;
	
	iSize = iNumberOfArrayElements * sizeof(float);
	cudaError_t result = cudaSuccess;
	
	//Code
	//Host memory allocation
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

	gold = (float*)malloc(iSize);
	if(gold == NULL)
	{
		printf("Host memory allocation is failed for gold array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}	
	
	//filling values into host arrays
	fillFloatArrayWithRandomNumbers(hostInput1,iNumberOfArrayElements);
	fillFloatArrayWithRandomNumbers(hostInput2,iNumberOfArrayElements);
	
	// device memory allocation
	
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

	//CUDA kernel configuration
	
	dim3 dimGrid = dim3((int)ceil((float)iNumberOfArrayElements / 256.0f),1,1);
	dim3 dimBlock = dim3(256,1,1);
	
	//CUDA kernel for vector addition
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	
	vecAddGPU <<<dimGrid,dimBlock>>>(deviceInput1,deviceInput2,deviceOutput,iNumberOfArrayElements);
	
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;
	
	//Copy data from device array into host array
	result = cudaMemcpy(hostOutput,deviceOutput,iSize,cudaMemcpyDeviceToHost);
	if(result != cudaSuccess)
	{
		printf("Device to host data copy is failed for hostOutput array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	//Vector addition on host
	vecAddCPU(hostInput1,hostInput2,gold,iNumberOfArrayElements);
	
	//Comparison
	const float epsilon = 0.000001f;
	
	int breakValue = -1,iCnt = 0;
	bool bAccuracy = true;
	for(iCnt = 0;iCnt < iNumberOfArrayElements;iCnt++)
	{
		float val1 = gold[iCnt];
		float val2 = hostOutput[iCnt];
		if(fabs(val1-val2) > epsilon)
		{
			bAccuracy = false;
			breakValue = iCnt;
			break;
		}
	}
	
	char str[128] = {'\0'};
	
	if(bAccuracy == false)
	{
		sprintf(str,"Comparison of CPU and GPU vector addition is not within accuracy of 0.000001 at array index %d\n",breakValue);
	}
	else
	{
		sprintf(str,"Comparison of CPU and GPU vector addition is not within accuracy of 0.000001\n");
	}
	
	//Output
	
	printf("\nArray1 begins from 0th index %.6f to %dth index %.6f\n",hostInput1[0],iNumberOfArrayElements - 1,hostInput1[iNumberOfArrayElements - 1]);
	
	printf("\nArray2 begins from 0th index %.6f to %dth index %.6f\n",hostInput2[0],iNumberOfArrayElements - 1,hostInput2[iNumberOfArrayElements - 1]);
	
	printf("\nCUDA kernel Grid dimension = %d,%d,%d and Block dimension = %d,%d,%d\n",dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);
	
	printf("\nOutput Array begins from 0th index %.6f to %dth index %.6f\n",hostOutput[0],iNumberOfArrayElements - 1,hostOutput[iNumberOfArrayElements - 1]);
	
	printf("\n===================================================================\n");
	printf("Time taken for vector addition on CPU = %.6f\n",timeOnCPU);
	printf("Time taken for vector addition on GPU = %.6f\n",timeOnGPU);
	printf("===================================================================\n");		
		
	cleanup();
	
	exit(EXIT_FAILURE);
	
}
