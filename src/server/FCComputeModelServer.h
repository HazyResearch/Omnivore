
#ifndef _FCCOMPUTEMODELSERVER_H
#define _FCCOMPUTEMODELSERVER_H

#include <iostream>
#include <string>
#include <cstring>
#include <glog/logging.h>

#include "server/Server.h"
#include "message/OmvMessage.h"
#include "../CaffeConTroll/src/DeepNet.h"

#include <zmq.h>

using namespace std;
using namespace libconfig;

class FCComputeModelServer : public Server{
public:

  string name;
  string bind;

  int nfloats;      // For input data and gradients (unlike ConvModelServer,
                    // where nfloats is model and model gradient size)

  // Unused now, can just get rid of dependency on libconfig++
  // using Server::input;
  // using Server::output;
  // using Server::models;

  FCComputeModelServer(string _name, string _bind) :
    name(_name), bind(_bind), nfloats(0) {}

  /**
   * A ConvModelServer does two things
   *   - listen to request that asks model.
   *   - listen to response that returns the gradient.
   **/
  void start(){
  
    LOG(INFO) << "Starting FCComputeModelServer[" << name << "]..." << std::endl;

    // -------------------------------------------------------------------------
    // Bind 
    // -------------------------------------------------------------------------
    void *context = zmq_ctx_new ();
    void *responder = zmq_socket (context, ZMQ_REP);
    int rc = zmq_bind (responder, bind.c_str());
    assert (rc == 0);
    LOG(INFO) << "Binded to " << bind << std::endl;

    // -------------------------------------------------------------------------
    // Read parameter files and construct network
    // -------------------------------------------------------------------------
    // SHADJIS TODO -- These will be created and passed in by scheduler (main.cpp or python)
    // Hard-code files for now. See comment in ConvModelServer.h.
    // Note: like the conv model server, this server does not need to read any labels or data from
    // the corpus (lmdb). The only information currently used from lmdb is the size of the 
    // first fc layer (stored in each datum).
    std::string solver_file = "protos/solver.fc_model_server.prototxt";
    std::string data_binary = "protos/dummy.bin";   // Empty file (no file needed, but prevents automatic binary creation)
    std::string output_model_file = "fc_model.bin";
    BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
    Corpus * const corpus = DeepNet::load_network(solver_file.c_str(), data_binary, solver_param, net_param, bridges, true);
    
    SoftmaxBridge * const softmax = (SoftmaxBridge *) bridges.back();
    LogicalCubeFloat * const labels = softmax->p_data_labels;
    
    // For printing information
    const int display_iter = solver_param.display();
    float loss = 0.;
    float accuracy = 0.;
    
    // -------------------------------------------------------------------------
    // Allocate buffers and create messages
    // -------------------------------------------------------------------------
    
    // Size of input data and input data gradients
    nfloats = bridges.front()->get_input_data_size();

    // Allocate buffer for incoming messages
    // This will be the output from ConvComputeServer
    // Allocate a factor of 2 extra for this although should not be needed
    // Also, for the incoming data, recall that we need the conv compute server
    // to pass labels.
    int incoming_data_buf_size = sizeof(OmvMessage) + 2*sizeof(float)*(nfloats + corpus->mini_batch_size);
    LOG(INFO) << "Allocating " << (1.0*incoming_data_buf_size/1024/1024) << " MB for incoming messages (input data)" << std::endl;
    char * incoming_data_buf = new char[incoming_data_buf_size];
    
    // Allocate buffer for gradients (outgoing message)
    // We respond with these gradients, so we know the size exactly
    int outgoing_data_grad_buf_size = sizeof(OmvMessage) + 1*sizeof(float)*nfloats;
    LOG(INFO) << "Allocating " << (1.0*outgoing_data_grad_buf_size/1024/1024) << " MB for outgoing data gradients" << std::endl;
    char * outgoing_data_grad_buf = new char[outgoing_data_grad_buf_size];

    // Create the response message for returning the data gradients
    OmvMessage * outgoing_msg_data_grad = reinterpret_cast<OmvMessage*>(outgoing_data_grad_buf);
    outgoing_msg_data_grad->msg_type = ANSWER_GRADIENT_OF_SENT_DATA;
    outgoing_msg_data_grad->nelem = nfloats;
    assert(outgoing_msg_data_grad->size() == outgoing_data_grad_buf_size);

    // -------------------------------------------------------------------------
    // Main Loop
    // -------------------------------------------------------------------------
    int batch = 0;
    while (1) {

      // -----------------------------------------------------------------------
      // Wait for a message containing new data. Read it into incoming_data_buf.
      // -----------------------------------------------------------------------
      zmq_recv(responder, incoming_data_buf, incoming_data_buf_size, 0);
      // Create the message from this incoming buffer
      OmvMessage * incoming_msg_data = reinterpret_cast<OmvMessage*>(incoming_data_buf);
      assert(incoming_msg_data->msg_type == ASK_GRADIENT_OF_SENT_DATA);
      assert(incoming_msg_data->size() == int(sizeof(OmvMessage) + 1*sizeof(float)*(nfloats + corpus->mini_batch_size)));

      // Now answer the request for data gradients
      // This involves running FW and BW and returning the gradients
      // LOG(INFO) << "Responding to ASK_GRADIENT_OF_SENT_DATA Request" << endl;
  
      // -----------------------------------------------------------------------
      // Update input layer to point to the incoming batch of data
      // -----------------------------------------------------------------------
      // This is same as switching from training to validation set, and just 
      // requires updating bridge 0 input data to point to incoming_msg_data->content.
      // Note: The implementation for this is to have the scheduler create a 
      // train proto file that makes a data layer of the correct size to match
      // the fc layer input.
      // Note that we are also ensuring the first layer's data is on the CPU,
      // which is awlays true when the first layer is the data layer.
      // If however in the future we want direct copy from GPU on conv compute
      // server to GPU on fc compute model server, we need to change this.
      //
      // SHADJIS TODO: I want to do this assertion: 
      //   assert(!bridges[0]->get_share_pointer_with_prev_bridge());
      // since the first bridge has no previous bridge, i.e. it should not share
      // any pointer with the previous bridge. However, for the first layer this
      // is set to true for networks not using GPUs because the current check sees 
      // that neither this layer nor the previous (non-existent) use any GPUs. So I
      // will omit this assertion, since it is always true that the first layer
      // (data input) is on the host, but if in the future we want to directly
      // update device memory (e.g. direct copy from one server to another) we
      // need to adjust the update_p_input_layer_data_CPU_ONLY call to pass in a
      // device memory pointer.
      // Note: we pass incoming_msg_data->content + corpus->mini_batch_size since
      // the convention we use now is to pass the labels first, then the data.
      bridges[0]->update_p_input_layer_data_CPU_ONLY(incoming_msg_data->content + corpus->mini_batch_size);
      assert(bridges[0]->p_input_layer->p_data_cube->get_p_data() == incoming_msg_data->content + corpus->mini_batch_size);
      // Debug
      // cout << incoming_msg_data->content[0] << endl;    // Print the first element of the data
  
      // -----------------------------------------------------------------------
      // Read in the next mini-batch labels from the sent message
      // -----------------------------------------------------------------------
      
      // Initialize labels for this mini batch
      labels->set_p_data(incoming_msg_data->content);
      
      // -----------------------------------------------------------------------
      // Run forward and backward pass
      // -----------------------------------------------------------------------

      softmax->reset_loss();
      
      // Forward pass
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        (*bridge)->forward();
      }

      loss += (softmax->get_loss() / float(corpus->mini_batch_size));
      accuracy += float(DeepNet::find_accuracy(labels, (*--bridges.end())->p_output_layer->p_data_cube)) / float(corpus->mini_batch_size);

      // Backward pass
      for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
        (*bridge)->backward();
      }

      // Check if we should print batch status
      if ( (batch+1) % display_iter == 0 ) {
        float learning_rate = Util::get_learning_rate(solver_param.lr_policy(), solver_param.base_lr(), solver_param.gamma(),
          batch+1, solver_param.stepsize(), solver_param.power(), solver_param.max_iter());
        
        std::cout << "Training Status Report (Mini-batch iter " << batch << "), LR = " << learning_rate << std::endl;
        std::cout << "  \033[1;32m";
        std::cout << "Loss & Accuracy [Average of Past " << display_iter << " Iterations]\t" << loss/float(display_iter) << "\t" << float(accuracy)/(float(display_iter));
        std::cout << "\033[0m" << std::endl;
        loss = 0.;
        accuracy = 0.;
      }
            
      // -----------------------------------------------------------------------
      // Fill outgoing_msg_data_grad->content with the data gradients
      // -----------------------------------------------------------------------
      // SHADJIS TODO: See comment above, this code needs to change if doing a
      // direct copy across servers to GPU memory to use device memory pointers.
      //assert(!bridges[0]->get_share_pointer_with_prev_bridge());
      memcpy(outgoing_msg_data_grad->content, 
        bridges[0]->p_input_layer->p_gradient_cube->get_p_data(),
        sizeof(float)*nfloats);
      
      // -----------------------------------------------------------------------
      // Send the data gradients back
      // -----------------------------------------------------------------------
      // LOG(INFO) << "Sending ANSWER_GRADIENT_OF_SENT_DATA Response" << endl;
      zmq_send (responder, outgoing_msg_data_grad, outgoing_msg_data_grad->size(), 0);
      
      ++batch;
    }

    // -------------------------------------------------------------------------
    // Save model and destroy network
    // -------------------------------------------------------------------------
    
    // Save model to file unless snapshot_after_train was set to false
    if (solver_param.snapshot_after_train()) {
      DeepNet::write_model_to_file(bridges, output_model_file);
      std::cout << "\nTrained model written to " + output_model_file +  ".\n";
    } else {
      std::cout << "\nNot writing trained model to file (snapshot_after_train = false)\n";
    }

    // Destroy network
    DeepNet::clean_up(bridges, corpus);

  }
};

#endif

