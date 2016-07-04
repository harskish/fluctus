CXX = g++
FRAMEWORKS = -framework OpenCL -framework OpenGL
LIBS = -I./include -I/usr/local/include -L/usr/local/lib -lglfw3
CXXFLAGS = -std=c++0x -O3 -Wall -Werror
SRCDIR = src
SOURCES := $(wildcard $(SRCDIR)/*.cpp)

B_LNX_PKG = -L/usr/local/lib -lglfw3 -lrt -lm -ldl -lXrandr -lXinerama -lXxf86vm -lXext -lXcursor -lXrender -lXfixes -lX11 -lpthread -lxcb -lXau -lXdmcp
LNX_PKG = -L/usr/local/lib -lglfw3 -lrt -lm -ldl -lXrandr -lXinerama -lXxf86vm -lXext -lXcursor -lXrender -lXfixes -lX11 -lpthread -lxcb -lXau -lXdmcp

all:
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $(LIBS) $(SOURCES) -o main

linux:
	$(CXX) -std=c++0x -I/usr/local/include -I./include $(LNX_PKG) $(SOURCES) -o main

old:
	$(CXX) -std=c++0x -O3 `pkg-config --cflags glfw3` -I./include `pkg-config --static --libs glfw3` $(SOURCES) -o main

clean:
	rm -f *.o main
