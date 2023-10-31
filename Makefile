.PHONY: clean

auxiliary_funcs.so: auxiliary_funcs.c
	$(CC) -O3 -shared -o auxiliary_funcs.so -fPIC auxiliary_funcs.c

clean:
	rm auxiliary_funcs.so
