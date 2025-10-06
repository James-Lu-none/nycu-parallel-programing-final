# Note: The following CMake FetchContent commands are for CMake, not Makefile.
# If you want to fetch tracy using Makefile, you can use git clone as a step.

TRACY_DIR = tracy

.PHONY: tracy
tracy:
	git -C $(TRACY_DIR) pull || git clone --depth 1 https://github.com/wolfpld/tracy.git $(TRACY_DIR)
SRC = N_body.c $(TRACY_DIR)/public/TracyClient.cpp
OUT = N_body

CFLAGS = -Wall -O2 -Dtracy_enable=true -pg -g
LDFLAGS = $(shell sdl2-config --cflags --libs) -lm

all: $(OUT)

$(OUT): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT) *.o gmon.out