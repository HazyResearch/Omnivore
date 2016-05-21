
#ifndef _CONVMODELSERVER_H
#define _CONVMODELSERVER_H

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <glog/logging.h>

#include "server/Server.h"
#include "message/OmvMessage.h"
#include "../CaffeConTroll/src/DeepNet.h"

#include <zmq.h>
#include <mutex>

#include "broker/Broker_N_1.h"


class ConvModelServer : public Server{
public:

  // SHADJIS TODO: These 3 should be taken from the parent class with using
  std::string name;
  std::string solver_file;
  
  std::mutex hogwild_lock;
  std::vector <std::string> broadcast_ports;
  std::vector <std::string> listen_ports;

  int nfloats;  // For both model and gradient buffers
  int group_size;

  ConvModelServer(string _name, std::string _solver_file,
    int _groupsize, std::vector <std::string> _broadcast_ports, std::vector <std::string> _listen_ports) : 
    name(_name), solver_file(_solver_file),
    broadcast_ports(_broadcast_ports), listen_ports(_listen_ports),
    nfloats(0), group_size(_groupsize) {}

  /**
   * A ConvModelServer does two things
   *   - listen to request that asks model.
   *   - listen to response that returns the gradient.
   **/

  void start(){
  
    VLOG(2) << "Starting ConvModelServer[" << name << "]..." << std::endl;

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
    Corpus * const corpus = DeepNet::load_network(solver_file.c_str(), solver_param, net_param, bridges, true);
    // DeepNet::read_full_snapshot(bridges, "/home/software/dcct/experiments//solver_template_1mpg_OPTIMIZER_DECISION_No_Sched/server_input_files-2016-05-09-22-33-32/solver.conv_model_compute_server.prototxt.snapshot_iter100000");
    
    // SHADJIS TODO: Corpus is unused but the param files are used. We can parse those files without having to read the corpus.

    // Get the number of conv compute server groups
    assert(broadcast_ports.size() == listen_ports.size());
    const size_t num_groups = broadcast_ports.size();
    
    // Start a thread for each group
    const int snapshot_multiplier = 2 * DeepNet::get_num_model_bridges(bridges); // *2 since fw + bw pass;
    const int snapshot = solver_param.snapshot() * snapshot_multiplier;
    int batch = 0;
    std::vector<std::thread> threads;
    for (size_t thread_idx=0; thread_idx < num_groups; ++thread_idx) {
      threads.push_back(thread([&, thread_idx]() {
      
        // -------------------------------------------------------------------------
        // Allocate buffers and create messages
        // -------------------------------------------------------------------------
        
        // Size of model and model gradients
        nfloats = DeepNet::get_parameter_size(bridges);
        
        // Buffer for incoming messages is not needed because it is within each broker

        // Allocate buffer for model (outgoing message)
        // We respond with the model, so we know its size exactly
        int outgoing_model_buf_size = sizeof(OmvMessage) + 1*sizeof(float)*nfloats;
        VLOG(2) << "Thread " << thread_idx << " allocating " << (1.0*outgoing_model_buf_size/1024/1024) << " MB for outgoing model" << std::endl;
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
        // Create worker 
        // -------------------------------------------------------------------------
        auto UDF = [&](OmvMessage ** msgs, int nmsg, OmvMessage * & msg){
          OmvMessage * incoming_msg = msgs[0];
          // Answer request for the conv model
          hogwild_lock.lock();
          if(incoming_msg->msg_type == ASK_MODEL){
            VLOG(2) << "Responding to ASK_MODEL Request of BRIDGE " << incoming_msg->bridgeid << std::endl;
            size_t model_nelem = DeepNet::get_ith_models(bridges, outgoing_msg_send_master_model->content, incoming_msg->bridgeid);
            msg = outgoing_msg_send_master_model;
            msg->nelem = model_nelem;
            //std::cout << msg->msg_type << "   " << ANSWER_MODEL << std::endl;
          // Receive gradients and update model
          }else if(incoming_msg->msg_type == ASK_UPDATE_GRADIENT){
            VLOG(2) << "Responding to ASK_UPDATE_GRADIENT Request of BRIDGE " << incoming_msg->bridgeid << std::endl;
            // Now that we have all the gradients, we can either update multiple times (for each one)
            // or add them then do a single update. In order to preserve the statistical efficiency
            // in all cases, we will add first then update once.
            // SHADJIS TODO: I can make this faster by minimizing # writes the way we do in SplitBridge backward pass
            for(int i=1;i<nmsg;i++){
              for(int elem=0; elem < msgs[0]->nelem; ++elem){
                msgs[0]->content[elem] += msgs[i]->content[elem];
              }
            }
            DeepNet::update_ith_models_with_gradients(bridges, msgs[0]->content, incoming_msg->bridgeid);
            msg = &outgoing_msg_reply_update_gradient;
          }else{
            LOG(WARNING) << "Ignore unsupported message type " << incoming_msg->msg_type << std::endl;
          }          
          
          // Check if we should write a snapshot
          if (snapshot > 0 && (batch+1) % snapshot == 0) {
            DeepNet::write_full_snapshot(bridges, solver_file, (batch+1)/snapshot_multiplier);
          }
          
          ++ batch;
          hogwild_lock.unlock();
        };  // END UDF

        // Start this broker with the input ports
        Broker_N_1<decltype(UDF)> broker(listen_ports[thread_idx], broadcast_ports[thread_idx], outgoing_model_buf_size, outgoing_model_buf_size, group_size);
        broker.start(UDF);
      
      }));  // END THREAD LAMBDA
    }   // END LOOP OVER THREADS

    // Join
    for (size_t i=0; i < num_groups; ++i) {
      threads[i].join();
    }
  
    // -------------------------------------------------------------------------
    // Save model and destroy network
    // -------------------------------------------------------------------------
    
    // Save model to file unless snapshot_after_train was set to false
    if (solver_param.snapshot_after_train()) {
      DeepNet::write_model_to_file(bridges, output_model_file);
      std::cout << std::endl << "Trained model written to " << output_model_file << "." << std::endl;
    } else {
      std::cout << std::endl << "Not writing trained model to file (snapshot_after_train = false)" << std::endl;
    }

    // Clean up network
    DeepNet::clean_up(bridges, corpus);
  } // END START FUNCTION

};  // END CLASS

#endif

