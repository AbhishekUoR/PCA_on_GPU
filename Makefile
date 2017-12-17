all : gpu_pca 

SOURCES = gpu_pca.cu
HEADERS = helper_cuda.h helper_string.h Timer.h

gpu_pca	:	$(SOURCES) $(HEADERS) 
				nvcc --gpu-architecture=sm_35 \
				--compiler-options -Wall \
				$(SOURCES) -o $@ -lcusolver -lcurand -lcublas

.phony: clean
clean:
	rm -rf gpu_pca
