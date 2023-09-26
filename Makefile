# Compiler and compiler flags
CXX = g++
CXXFLAGS = -Wall -std=c++11

# Output executable
TARGET = main

# Object files
OBJS = utility.o test_functions.o optimization.o genetic.o main.o

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

utility.o: utility.cpp utility.h
	$(CXX) $(CXXFLAGS) -c utility.cpp

test_functions.o: test_functions.cpp test_functions.h utility.h
	$(CXX) $(CXXFLAGS) -c test_functions.cpp

optimization.o: optimization.cpp optimization.h test_functions.h utility.h
	$(CXX) $(CXXFLAGS) -c optimization.cpp

genetic.o: genetic.cpp genetic.h optimization.h test_functions.h utility.h
	$(CXX) $(CXXFLAGS) -c genetic.cpp

main.o: main.cpp genetic.h optimization.h test_functions.h utility.h
	$(CXX) $(CXXFLAGS) -c main.cpp

clean:
	rm -rf *.o $(TARGET)
