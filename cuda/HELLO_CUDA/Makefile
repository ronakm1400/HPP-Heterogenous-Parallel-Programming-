all: compile	

Shell:
	 @ (/bin/sh)
	
compile:
	@ (nvcc HelloCUDA.cu -o HelloCuda)
	@ (echo "Program compiled successfully")

run:
	@ (./HelloCuda)
	
clean:
	@ (rm HelloCuda)		
	@ (echo "File removed")
