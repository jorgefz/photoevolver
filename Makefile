

BASEDIR=$(PWD)/photoevolver/libcfn

build: $(BASEDIR)/src/constants.h $(BASEDIR)/src/mloss.c $(BASEDIR)/src/struct.c
	$(CC) -Wall -Wextra -O2 -shared -o $(BASEDIR)/shared/libcfn.so -fPIC $(BASEDIR)/src/mloss.c $(BASEDIR)/src/struct.c 	

clean:
	rm -f $(BASEDIR)/shared/*

