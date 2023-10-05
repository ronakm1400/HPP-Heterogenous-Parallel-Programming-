all: compile	

Shell:
	 @ (/bin/sh)
	
compile:
	@ (nvcc DevProperties.cu -o devprop)
	@ (echo "Program compiled successfully")

run:
	@ (./devprop)
	
clean:
	@ (rm devprop)		
	@ (echo "File removed")
