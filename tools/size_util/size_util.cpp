// This file loads a network and prints out the output size of its final layer

#include <iostream>
#include <cstdlib>
#include <assert.h>
#include "../../CaffeConTroll/src/DeepNet.h"

int main(int argc, char **argv)
{
    assert(argc == 2);
    char * solver_file = argv[1];
    BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
    Corpus * const corpus = DeepNet::load_network(solver_file, solver_param, net_param, bridges, true);
    // Size of the last layer (number of floats per image going into first fc layer)
    size_t nfloats_output_data = bridges.back()->get_output_data_size();
    std::cout << nfloats_output_data / corpus->mini_batch_size << "\n";

    return 0;
}
