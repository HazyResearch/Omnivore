
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
#include <mutex>

#include "broker/Broker_N_1.h"


void UDF (OmvMessage ** msgs, int nmsg, OmvMessage * msg, void * scratch){
  msg->nelem = 1;
  msg->content[0] = 1024;
  msg->content[1] = 2048;
}

class ConvModelServer : public Server{
public:

  // SHADJIS TODO: These 3 should be taken from the parent class with using
  std::string name;
  std::string solver_file;
  std::string data_binary;
  
  std::string bind;

  int nfloats;  // For both model and gradient buffers

  ConvModelServer(string _name, std::string _bind, std::string _solver_file, std::string _data_binary) : 
    name(_name), solver_file(_solver_file), data_binary(_data_binary),
    bind(_bind), nfloats(0) {}

  /**
   * A ConvModelServer does two things
   *   - listen to request that asks model.
   *   - listen to response that returns the gradient.
   **/
  void start(){
  
    LOG(INFO) << "Starting ConvModelServer[" << name << "]..." << std::endl;

    // -------------------------------------------------------------------------
    // Read parameter files and construct network
    // -------------------------------------------------------------------------
    // Updating the gradient requires some parameters read from protobuf
    // files. Read these files in here. This is similar to DeepNet.h load_and_train_network.

    // SHADJIS TODO: Here we construct the entire network but we don't need to because
    // this server only needs the models and learning rates / regularization per layer.
    // However the data is small so this does not matter. The LMDB is also unused except
    // for the size of the input layer which is stored in each datum.
    std::string output_model_file = "conv_model.bin";
    BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
    Corpus * const corpus = DeepNet::load_network(solver_file.c_str(), data_binary.c_str(), solver_param, net_param, bridges, true);
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

    std::mutex hogwild_lock;

    // -------------------------------------------------------------------------
    // Create worker 
    // -------------------------------------------------------------------------
    auto UDF = [&](OmvMessage ** msgs, int nmsg, OmvMessage * & msg){
      OmvMessage * incoming_msg = msgs[0];
      // Answer request for the conv model
      hogwild_lock.lock();
      if(incoming_msg->msg_type == ASK_MODEL){
        LOG(INFO) << "Responding to ASK_MODEL Request of BRIDGE " << incoming_msg->bridgeid << std::endl;
        DeepNet::get_ith_models(bridges, outgoing_msg_send_master_model->content, incoming_msg->bridgeid);
        msg = outgoing_msg_send_master_model;
        //std::cout << msg->msg_type << "   " << ANSWER_MODEL << std::endl;
      // Receive gradients and update model
      }else if(incoming_msg->msg_type == ASK_UPDATE_GRADIENT){
        LOG(INFO) << "Responding to ASK_UPDATE_GRADIENT Request of BRIDGE " << incoming_msg->bridgeid << std::endl;
        // CE TODO: Gradient should be divided by # workers in the group.
        for(int i=0;i<nmsg;i++){
          //DeepNet::update_all_models_with_gradients(bridges, msgs[i]->content);
          DeepNet::update_ith_models_with_gradients(bridges, msgs[i]->content, incoming_msg->bridgeid);
        }
        msg = &outgoing_msg_reply_update_gradient;
      }else{
        LOG(WARNING) << "Ignore unsupported message type " << incoming_msg->msg_type << std::endl;
      }
      hogwild_lock.unlock();
    };

    /********
     * TODO CE: THIS IS WHERE THE SCHEDULER COMES IN
     ********/
    int N_CONVSERVER_PER_GROUP=2;
    Broker_N_1<decltype(UDF)> broker("tcp://*:7555", "tcp://*:7556", outgoing_model_buf_size, outgoing_model_buf_size, N_CONVSERVER_PER_GROUP);

    auto start_broker = [&](Broker_N_1<decltype(UDF)> * _broker){
      _broker->start(UDF);
    };
    std::thread thread1(start_broker, &broker);
    thread1.join();



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

