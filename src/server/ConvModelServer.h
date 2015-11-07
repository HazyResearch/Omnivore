
#ifndef _CONVMODELSERVER_H
#define _CONVMODELSERVER_H

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

class ConvModelServer : public Server{
public:

  string name;
  string bind;

  int nfloats;  // For both model and gradient buffers

  // Unused now, can just get rid of dependency on libconfig++
  // using Server::input;
  // using Server::output;
  // using Server::models;

  ConvModelServer(string _name, string _bind) : 
    name(_name), bind(_bind), nfloats(0) {}

  /**
   * A ConvModelServer does two things
   *   - listen to request that asks model.
   *   - listen to response that returns the gradient.
   **/
  void start(){
  
    LOG(INFO) << "Starting ConvModelServer[" << name << "]..." << std::endl;

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
    // Updating the gradient requires some parameters read from protobuf
    // files. Read these files in here. This is similar to DeepNet.h load_and_train_network.

    // SHADJIS TODO: Here we construct the entire network but we don't need to because
    // this server only needs the models and learning rates / regularization per layer.
    // However the data is small so this does not matter.
    // SHADJIS TODO -- These will be created and passed in by scheduler (main.cpp or python)
    std::string solver_file = "protos/solver.conv_model_server.prototxt";
    std::string data_binary = "protos/dummy.bin";   // Empty file (no file needed, but prevents automatic binary creation)
    std::string output_model_file = "conv_model.bin";
    BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
    Corpus * const corpus = DeepNet::load_network(solver_file.c_str(), data_binary, solver_param, net_param, bridges, true);
    // SHADJIS TODO: Corpus is unused but the param files are used. We can parse those files without having to read the corpus.

    // -------------------------------------------------------------------------
    // Allocate buffers and create messages
    // -------------------------------------------------------------------------
    
    // Size of model and model gradients
    nfloats = DeepNet::get_parameter_size(bridges);
    
    // Allocate buffer for incoming messages
    // This will either be an empty request for a model, or a large
    // request sending back gradients
    // Allocate a factor of 2 extra for this although should not be needed
    int incoming_buf_size = sizeof(OmvMessage) + 2*sizeof(float)*nfloats;
    LOG(INFO) << "Allocating " << (1.0*incoming_buf_size/1024/1024) << " MB for incoming messages (model gradients)" << std::endl;
    char * incoming_buf = new char[incoming_buf_size];

    // Allocate buffer for model (outgoing message)
    // We respond with the model, so we know its size exactly
    int outgoing_model_buf_size = sizeof(OmvMessage) + 1*sizeof(float)*nfloats;
    LOG(INFO) << "Allocating " << (1.0*outgoing_model_buf_size/1024/1024) << " MB for outgoing model" << std::endl;
    char * outgoing_model_buf = new char[outgoing_model_buf_size];

    // Create the response message for returning the model
    OmvMessage * outgoing_msg_send_master_model = reinterpret_cast<OmvMessage*>(outgoing_model_buf);
    outgoing_msg_send_master_model->msg_type = ANSWER_MODEL;
    outgoing_msg_send_master_model->nelem = nfloats;
    assert(outgoing_msg_send_master_model->size() == outgoing_model_buf_size);
    
    // Create the response message for acknowledging the gradient is updated
    OmvMessage outgoing_msg_reply_update_gradient;
    outgoing_msg_reply_update_gradient.msg_type = ANSWER_UPDATE_GRADIENT;
    outgoing_msg_reply_update_gradient.nelem = 0;
    assert(outgoing_msg_reply_update_gradient.size() == sizeof(OmvMessage));

    // -------------------------------------------------------------------------
    // Main Loop
    // -------------------------------------------------------------------------
    while (1) {
    
      // Wait for a message. Read it into incoming_buf.
      // This will either be an empty request for a model, or a large
      // request sending back gradients
      zmq_recv(responder, incoming_buf, incoming_buf_size, 0);
      // Create the message from this incoming buffer
      OmvMessage * incoming_msg = reinterpret_cast<OmvMessage*>(incoming_buf);
      
      // Answer request for the conv model
      if(incoming_msg->msg_type == ASK_MODEL){
      
        assert(incoming_msg->size() == sizeof(OmvMessage));
        
        // Reply Current Model
        LOG(INFO) << "Responding to ASK_MODEL Request" << endl;
        // Load the model
        LOG(INFO) << "    Loading the model" << endl;
        DeepNet::get_all_models(bridges, outgoing_msg_send_master_model->content);
        // Send this model object back
        LOG(INFO) << "Sending ANSWER_MODEL Response" << endl;
        zmq_send (responder, outgoing_msg_send_master_model, outgoing_msg_send_master_model->size(), 0);
        
      // Receive gradients and update model
      }else if(incoming_msg->msg_type == ASK_UPDATE_GRADIENT){
      
        assert(incoming_msg->size() == outgoing_model_buf_size);
        
        LOG(INFO) << "Responding to ASK_UPDATE_GRADIENT Request" << endl;
        // Update gradients and send acknowledgement
        LOG(INFO) << "    Updating Gradient" << endl;
        // Call CcT to update the model given this model gradient.
        DeepNet::update_all_models_with_gradients(bridges, incoming_msg->content);
        // Send back an acknowledgement that the gradient has been updated
        LOG(INFO) << "Sending ANSWER_UPDATE_GRADIENT Response" << endl;
        zmq_send (responder, &outgoing_msg_reply_update_gradient, outgoing_msg_reply_update_gradient.size(), 0);
        
      }else{
        LOG(WARNING) << "Ignore unsupported message type " << incoming_msg->msg_type << endl;
      }
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

    // Clean up network
    DeepNet::clean_up(bridges, corpus);

  }
};

#endif

