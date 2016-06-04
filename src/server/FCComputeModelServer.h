
#ifndef _FCCOMPUTEMODELSERVER_H
#define _FCCOMPUTEMODELSERVER_H

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


class FCComputeModelServer : public Server{
public:

  // SHADJIS TODO: These 3 should be taken from the parent class with using
  std::string name;
  std::string solver_file;
  
  int nfloats;      // For input data and gradients (unlike ConvModelServer,
                    // where nfloats is model and model gradient size)
  int group_size;
  
  std::mutex hogwild_lock;
  std::vector <std::string> broadcast_ports;
  std::vector <std::string> listen_ports;

  FCComputeModelServer(string _name, std::string _solver_file,
    int _groupsize, std::vector <std::string> _broadcast_ports, std::vector <std::string> _listen_ports) : 
    name(_name), solver_file(_solver_file),
    nfloats(0), group_size(_groupsize),
    broadcast_ports(_broadcast_ports), listen_ports(_listen_ports) {}

  /**
   * An FCComputeModelServer does two things
   *   - listen to request that asks model.
   *   - listen to response that returns the gradient.
   **/

  void start(){
  
    VLOG(2) << "Starting FCComputeModelServer[" << name << "]..." << std::endl;

    // -------------------------------------------------------------------------
    // Read parameter files and construct network
    // -------------------------------------------------------------------------
    // Note: like the conv model server, this server does not need to read any labels or data from
    // the corpus (lmdb). The only information currently used from lmdb is the size of the 
    // first fc layer (stored in each datum).
    std::string output_model_file = "fc_model.bin";
    BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
    Corpus * const corpus = DeepNet::load_network(solver_file.c_str(), solver_param, net_param, bridges, true);
    // DeepNet::read_full_snapshot(bridges, "/home/software/dcct/experiments//solver_template_1mpg_OPTIMIZER_DECISION_No_Sched/server_input_files-2016-05-09-22-33-32/solver.fc_model_compute_server.prototxt.snapshot_iter100000");
    
    SoftmaxBridge * const softmax = (SoftmaxBridge *) bridges.back();
    LogicalCubeFloat * const labels = softmax->p_data_labels;
    
    // For printing information
    const int display_iter = solver_param.display();
    const int snapshot = solver_param.snapshot();
    float loss = 0.;
    float accuracy = 0.;
  
    // Get the number of conv compute server groups
    assert(broadcast_ports.size() == listen_ports.size());
    const size_t num_groups = broadcast_ports.size();
    
    // Start a thread for each group
    int batch = 0;
    Timer timer;
    std::vector<std::thread> threads;
    for (size_t thread_idx=0; thread_idx < num_groups; ++thread_idx) {
      threads.push_back(thread([&, thread_idx]() {
      
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
        // This one still used (unlike conv model) since we want to reformat data
        int incoming_data_buf_size = sizeof(OmvMessage) + 2*sizeof(float)*(nfloats + corpus->mini_batch_size);
        VLOG(2) << "Thread " << thread_idx << " allocating " << (1.0*incoming_data_buf_size/1024/1024) << " MB for incoming messages (input data)" << std::endl;
        char * incoming_data_buf = new char[incoming_data_buf_size];
        
        // Allocate buffer for gradients (outgoing message)
        // We respond with these gradients, so we know the size exactly
        int outgoing_data_grad_buf_size = sizeof(OmvMessage) + 1*sizeof(float)*nfloats;
        VLOG(2) << "Thread " << thread_idx << " allocating " << (1.0*outgoing_data_grad_buf_size/1024/1024) << " MB for outgoing data gradients" << std::endl;
        char * outgoing_data_grad_buf = new char[outgoing_data_grad_buf_size];

        // Create the response message for returning the data gradients
        OmvMessage * outgoing_msg_data_grad = reinterpret_cast<OmvMessage*>(outgoing_data_grad_buf);
        outgoing_msg_data_grad->msg_type = ANSWER_GRADIENT_OF_SENT_DATA;
        outgoing_msg_data_grad->nelem = nfloats;
        assert(outgoing_msg_data_grad->size() == outgoing_data_grad_buf_size);

        // -------------------------------------------------------------------------
        // Create worker 
        // -------------------------------------------------------------------------
        auto UDF = [&](OmvMessage ** msgs, int nmsg, OmvMessage * & msg){

          hogwild_lock.lock();

          OmvMessage * incoming_msg_data = reinterpret_cast<OmvMessage *>(incoming_data_buf);

          const int nlabel_per_msg = corpus->mini_batch_size/nmsg;
          //std::cout << "----" << nlabel_per_msg << std::endl;
          //std::cout << "----" << nmsg << std::endl;
          const int ndata_per_msg  = msgs[0]->nelem - nlabel_per_msg;
          //std::cout << "----" << ndata_per_msg << std::endl;
          //std::cout << "~~~~" << incoming_msg_data << std::endl;

          // Init incoming_msg_data from different inputs.
          incoming_msg_data->msg_type = msgs[0]->msg_type;

          //std::cout << "******" << std::endl;

          incoming_msg_data->nelem = (ndata_per_msg+nlabel_per_msg)*nmsg;

          //std::cout << "******" << std::endl;

          for(int i=0;i<nmsg;i++){
            //std::cout << "~~~~" << i << " / " << nmsg << std::endl;
            memcpy(&incoming_msg_data->content[i*nlabel_per_msg], msgs[i]->content, sizeof(float)*nlabel_per_msg);
            //std::cout << "####" << std::endl;
            memcpy(&incoming_msg_data->content[corpus->mini_batch_size + i*ndata_per_msg], 
                    msgs[i]->content + nlabel_per_msg, 
                    sizeof(float)*ndata_per_msg);
            // std::cout << "%%%%%%" << std::endl;
          }

          assert(incoming_msg_data->msg_type == ASK_GRADIENT_OF_SENT_DATA);
          assert(incoming_msg_data->size() == int(sizeof(OmvMessage) + 1*sizeof(float)*(nfloats + corpus->mini_batch_size)));

          // -----------------------------------------------------------------------
          // Update input layer to point to the incoming batch of data
          // -----------------------------------------------------------------------
          VLOG(2) << "~~~~ ENTER STATE Update input layer" << std::endl;
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
          VLOG(2) << "~~~~ EXIT STATE Update input layer" << std::endl;
      
          // -----------------------------------------------------------------------
          // Read in the next mini-batch labels from the sent message
          // -----------------------------------------------------------------------
          
          // Initialize labels for this mini batch
          labels->set_p_data(incoming_msg_data->content);
          
          // -----------------------------------------------------------------------
          // Run forward and backward pass
          // -----------------------------------------------------------------------

          VLOG(2) << "~~~~ ENTER STATE FC FW" << std::endl;
          softmax->reset_loss();
          
          // Forward pass
          DeepNet::run_forward_pass(bridges);
          VLOG(2) << "~~~~ EXIT STATE FC FW" << std::endl;

          VLOG(2) << "~~~~ ENTER STATE ACC" << std::endl;
          loss += (softmax->get_loss() / float(corpus->mini_batch_size));
          accuracy += float(DeepNet::find_accuracy(labels, (*--bridges.end())->p_output_layer->p_data_cube)) / float(corpus->mini_batch_size);
          VLOG(2) << "~~~~ EXIT STATE ACC" << std::endl;

          // Backward pass
          VLOG(2) << "~~~~ ENTER STATE FC BW" << std::endl;
          DeepNet::run_backward_pass(bridges);

          // Check if we should print batch status
          if ( (batch+1) % display_iter == 0 ) {
            float learning_rate = Util::get_learning_rate(solver_param.lr_policy(), solver_param.base_lr(), solver_param.gamma(),
              batch+1, solver_param.stepsize(), solver_param.power(), solver_param.max_iter());
            std::cout << batch+1 << "\t" << timer.elapsed() << "\t" << loss/float(display_iter) << "\t" << float(accuracy)/(float(display_iter)) << std::endl;
            loss = 0.;
            accuracy = 0.;
          }
          VLOG(2) << "~~~~ EXIT STATE FC BW" << std::endl;
       
          // -----------------------------------------------------------------------
          // Fill outgoing_msg_data_grad->content with the data gradients
          // -----------------------------------------------------------------------
          // SHADJIS TODO: See comment above, this code needs to change if doing a
          // direct copy across servers to GPU memory to use device memory pointers.
          //assert(!bridges[0]->get_share_pointer_with_prev_bridge());
          VLOG(2) << "~~~~ ENTER STATE FC Get Grad" << std::endl;
          
          memcpy(outgoing_msg_data_grad->content, 
            bridges[0]->p_input_layer->p_gradient_cube->get_p_data(),
            sizeof(float)*nfloats);
          VLOG(2) << "~~~~ EXIT STATE FC Get Grad" << std::endl;
        
          msg = outgoing_msg_data_grad;
        
          // Check if we should write a snapshot
          if (snapshot > 0 && (batch+1) % snapshot == 0) {
            DeepNet::write_full_snapshot(bridges, solver_file, batch+1);
          }
        
          ++ batch;

          hogwild_lock.unlock();
        };  // END UDF

        // Start this broker with the input ports
        Broker_N_1<decltype(UDF)> broker(listen_ports[thread_idx], broadcast_ports[thread_idx], incoming_data_buf_size, outgoing_data_grad_buf_size, group_size);
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

