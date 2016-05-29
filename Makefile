CXX = g++
FRAMEWORKS = -framework OpenCL -framework OpenGL
LIBS = -I./include -I/usr/local/include -L/usr/local/lib -lglfw3
CXXFLAGS = -std=c++0x -O3 -Wall -Werror
SRCDIR = src
SOURCES := $(wildcard $(SRCDIR)/*.cpp)

all:
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $(LIBS) $(SOURCES) -o main

clean:
	rm -f *.o main
