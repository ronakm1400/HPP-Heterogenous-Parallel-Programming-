all: compile	

Shell:
	 @ (/bin/sh)
	
compile:
	@ (nvcc VecAddCUDA.cu -o VecAdd)
	@ (echo "Program compiled successfully")

run:
	@ (./VecAdd)
	
clean:
	@ (rm VecAdd)		
	@ (echo "File removed")
