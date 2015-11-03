
all:
	clang++ -Ofast -I./src src/main.cpp -lconfig++ -lglog -lzmq -o dcct