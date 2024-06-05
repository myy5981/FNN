.DEFAULT_GOAL := app

CC = gcc -m64
LINK = gcc -m64
AR = ar
CLEVEL = -O2
CFLAGS = -Wall -Werror $(CLEVEL)

OBJ = obj/matrix.o \
		obj/neuron.o \
		obj/relu.o \
		obj/softmax.o \
		obj/network.o \
		obj/mean_square.o \
		obj/corss_entropy.o \
		obj/random.o

TEST_EXE = test/matrix_test

.PHONY: app
app: mkdir bin/train bin/demo

.PHONY: test
test: mkdir bin/libfnn.so $(TEST_EXE) 

bin/demo: obj/demo.o bin/libfnn.so
	$(LINK) -L./bin $< -o $@ -lSDL2 -lfnn

bin/train: obj/train.o bin/libfnn.so
	$(LINK) -L./bin $< -o $@ -lfnn

bin/libfnn.so: $(OBJ)
	$(LINK) -shared $^ -o $@ -lm

obj/%.o: src/%.c
	$(CC) -c $(CFLAGS) $^ -o $@

test/%_test: test/%_test.c
	$(CC) $(CFLAGS) -I./src -L./bin $^ -o $@ -lfnn
	@echo ==========$@==========
	@./$@

.PHONY: clean
clean: 
	rm -rf bin obj test/*_test

.PHONY: mkdir
mkdir: 
	@mkdir -p obj bin