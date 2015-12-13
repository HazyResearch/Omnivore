all:
	clang++ -I./src src/test_client.cpp -o client -lzmq -mavx2 -msse2 
	clang++ -I./src src/test_server.cpp -o server -lzmq -mavx2 -msse2 