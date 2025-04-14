.DEFAULT := core_info

.PHONY: core_info
core_info:
	rm -f libcore_info.so
	gcc -fPIC -O3 -shared core_info.c -o libcore_info.so
