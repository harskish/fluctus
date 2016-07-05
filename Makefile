CXX = g++
FRAMEWORKS = -framework OpenCL -framework OpenGL
LIBS = -I./include -I/usr/local/include -L/usr/local/lib -lglfw3
CXXFLAGS = -std=c++0x -O3 -Wall -Werror
SRCDIR = src
SOURCES := $(wildcard $(SRCDIR)/*.cpp)

LNX_PKG = -L/usr/local/lib -L/usr/local/cuda-8.0/lib64/ -lglfw3 -pthread -lGLEW -lGLU -lGL -lrt -lXrandr -lXxf86vm -lXi -lXinerama -lX11 -ldl -lXinerama -lXcursor -lOpenCL

all:
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $(LIBS) $(SOURCES) -o main

linux:
	$(CXX) -std=c++0x -I/usr/local/include -I./include $(SOURCES) -o main $(LNX_PKG)

clean:
	rm -f *.o main