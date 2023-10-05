This HelloCUDA.cu Application is an example of First CUDA code where we perform addition between two float numbers on GPU.

We can compile this application as:
	nvcc HelloCUDA.c -o HelloCuda
	
To run this code we use below command:
	./HelloCuda
	
To perform this above tasks we have created makefile and have written the above commands in that file as it is.

all is the target in this makefile.(Which compiles the code.)

To run Makefile please enter below command:
make

The above command will execute the command which is been written in all targets inside Makefile.
In our case the above command will compile the code and create an Executable_File(exe).

To run exe please enter below command :
make run.

To delete the exe please enter below command :
make clean.
