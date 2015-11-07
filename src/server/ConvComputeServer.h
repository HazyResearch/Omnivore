
#ifndef _CONVCOMPUTESERVER_H
#define _CONVCOMPUTESERVER_H

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

class ConvComputeServer : public Server{
public:

  string name;
  string conv_bind;
  string fc_bind;

  // There are 4 buffers needed:
  // - The model and model gradients (these two are the same size)
  // - The output layer data and output layer gradients (these two are the same size)
  int nfloats_model;
  int nfloats_output_data;

  // Unused now, can just get rid of dependency on libconfig++
  // using Server::input;
  // using Server::output;
  // using Server::models;

  ConvComputeServer(string _name, string _conv_bind, string _fc_bind) : 
    name(_name), conv_bind(_conv_bind), fc_bind(_fc_bind), nfloats_model(0),
    nfloats_output_data(0) {}

  void start(){
  
    LOG(INFO) << "Starting ConvComputeServer[" << name << "]..." << std::endl;

    // -------------------------------------------------------------------------
    // Bind to both servers
    // -------------------------------------------------------------------------
    
    // This server will send 2 messages: sending gradients back to conv server, and sending
    // outputs to fc server. It will also receive a new model (same size as gradients it sends).
    
    void *context = zmq_ctx_new ();
    void *requester = zmq_socket (context, ZMQ_REQ);
    zmq_connect (requester, conv_bind.c_str());
    LOG(INFO) << "Binded to " << conv_bind << std::endl;

    void *context_fc = zmq_ctx_new ();
    void *requester_fc = zmq_socket (context_fc, ZMQ_REQ);
    zmq_connect (requester_fc, fc_bind.c_str());
    LOG(INFO) << "Binded to " << fc_bind << std::endl;

    // -------------------------------------------------------------------------
    // Read parameter files and construct network
    // -------------------------------------------------------------------------
    // SHADJIS TODO -- These will be created and passed in by scheduler (main.cpp or python)
    std::string solver_file = "protos/solver.conv_compute_server.prototxt";
    std::string data_binary = "/lfs/raiders1/0/shadjis/cct_tests/8_train_NEW.bin";
    BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
    // Modify all bridges to not update model gradients in backward pass (saves time)
    for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
      (*bridge)->set_update_model_gradients(false);
    }
    Corpus * const corpus = DeepNet::load_network(solver_file.c_str(), data_binary, solver_param, net_param, bridges, true);
    // Open the file for the first time during training
    FILE * pFile = fopen (corpus->filename.c_str(), "rb");
    if (!pFile)
      throw std::runtime_error("Error opening the corpus file: " + corpus->filename);
    // Keep track of the image number in the dataset we are on
    size_t current_image_location_in_dataset = 0;
    // size_t current_epoch = 0;    

    // -------------------------------------------------------------------------
    // Allocate buffers and create messages
    // -------------------------------------------------------------------------
    
    // Size of model
    nfloats_model = DeepNet::get_parameter_size(bridges);
    // Size of the last layer
    nfloats_output_data = bridges.back()->get_output_data_size();
    // std::cout << "\n#floats to put in fc proto = " << nfloats_output_data / corpus->mini_batch_size << "\n\n";

    // -------------------------------------------
    // Allocate buffer for outgoing messages
    // -------------------------------------------
    // The message objects for each buffer are created right below
    
    // Outgoing model gradients
    // assert(sizeof(OmvMessage) == 8); // Eventually we should assert the zero-length array has size 0 bytes
                                        // If not the buffer sizes below might be off by 1
    int outgoing_model_grad_buf_size = sizeof(OmvMessage) + 1*sizeof(float)*nfloats_model;
    LOG(INFO) << "Allocating " << (1.0*outgoing_model_grad_buf_size/1024/1024) << " MB for outgoing model gradients" << std::endl;
    char * outgoing_model_grad_buf = new char[outgoing_model_grad_buf_size];
    
    // Outgoing data
    // Recall that we need to pass labels as well.
    int outgoing_data_buf_size = sizeof(OmvMessage) + 1*sizeof(float)*(nfloats_output_data + corpus->mini_batch_size);
    LOG(INFO) << "Allocating " << (1.0*outgoing_data_buf_size/1024/1024) << " MB for outgoing data" << std::endl;
    char * outgoing_data_buf = new char[outgoing_data_buf_size];

    
    // Outgoing message which sends the model gradients
    OmvMessage * outgoing_msg_send_model_grad = reinterpret_cast<OmvMessage*>(outgoing_model_grad_buf);
    outgoing_msg_send_model_grad->msg_type = ASK_UPDATE_GRADIENT;
    outgoing_msg_send_model_grad->nelem = nfloats_model;
    assert(outgoing_msg_send_model_grad->size() == outgoing_model_grad_buf_size);

    // Outgoing message which asks for new model
    // Note: This could have been combined with outgoing_msg_send_model_grad, 
    // i.e. seinding gradients implies asking for new model (like the
    // message below outgoing_msg_send_data_and_ask_grad which does both)
    OmvMessage outgoing_msg_ask_model;
    outgoing_msg_ask_model.msg_type = ASK_MODEL;
    outgoing_msg_ask_model.nelem = 0;
    assert(outgoing_msg_ask_model.size() == sizeof(OmvMessage));

    // Outgoing message which sends the data and asks for gradients
    OmvMessage * outgoing_msg_send_data_and_ask_grad = reinterpret_cast<OmvMessage*>(outgoing_data_buf);
    outgoing_msg_send_data_and_ask_grad->msg_type = ASK_GRADIENT_OF_SENT_DATA;
    outgoing_msg_send_data_and_ask_grad->nelem = nfloats_output_data + corpus->mini_batch_size;
    assert(outgoing_msg_send_data_and_ask_grad->size() == outgoing_data_buf_size);
    
    // -------------------------------------------
    // Allocate buffer for incoming messages
    // -------------------------------------------
    // Allocate a factor of 2 extra for these although should not be needed
    // The message objects themselves for these 2 will be created inside the loop each iteration
    
    // Incoming model
    int incoming_model_buf_size = sizeof(OmvMessage) + 2*sizeof(float)*nfloats_model;
    LOG(INFO) << "Allocating " << (1.0*incoming_model_buf_size/1024/1024) << " MB for incoming model" << std::endl;
    char * incoming_model_buf = new char[incoming_model_buf_size];

    // Incoming data gradients
    int incoming_data_grad_buf_size = sizeof(OmvMessage) + 2*sizeof(float)*nfloats_output_data;
    LOG(INFO) << "Allocating " << (1.0*incoming_data_grad_buf_size/1024/1024) << " MB for incoming data gradients" << std::endl;
    char * incoming_data_grad_buf = new char[incoming_data_grad_buf_size];

    // Incoming message which acknowledges that the model gradients we
    // sent have finished updating, and therefore we can ask for a new model
    // Note: This could have been combined into outgoing_msg_send_model_grad,
    // i.e. sending a model gradient implies waiting for a new model
    OmvMessage incoming_msg_model_grad_updated;
    incoming_msg_model_grad_updated.msg_type = ANSWER_UPDATE_GRADIENT; // This line not needed, in fact should probably not even be set
    incoming_msg_model_grad_updated.nelem = 0;
    assert(incoming_msg_model_grad_updated.size() == sizeof(OmvMessage));

    // -------------------------------------------------------------------------
    // Main Loop
    // -------------------------------------------------------------------------
    while(1){
    
      // -----------------------------------------------------------------------
      // Read in the next mini-batch from file and update labels from lmdb
      // -----------------------------------------------------------------------
      
      // Read in the next mini-batch from file
      size_t rs = fread(corpus->images->get_p_data(), sizeof(DataType_SFFloat), corpus->images->n_elements, pFile);
      
      // If we read less than we expected, read the rest from the beginning
      size_t num_floats_left_to_read = corpus->images->n_elements - rs;
      if (num_floats_left_to_read > 0) {
      
        // Close the file and re-open it
        fclose(pFile);
        pFile = fopen (corpus->filename.c_str(), "rb");
        if (!pFile)
          throw std::runtime_error("Error opening the corpus file: " + corpus->filename);
          
        // Read the remaining data from the file, adjusting the pointer to where we
        // read until previously as well as the amount to read
        size_t rs2 = fread((float *) (corpus->images->get_p_data()) + rs, sizeof(DataType_SFFloat), num_floats_left_to_read, pFile);
        assert(rs2 == num_floats_left_to_read);
        
        // Also, we need to copy over the labels to the outgoing message buffer.
        // The labels are all allocated in corpus->labels. Normally we just copy
        // from the corpus labels cube data, but since the labels we want are partly
        // at the end of that array and partly at the beginning, we have to do 2 copies
        
        // Check if we actually read nothing (i.e. we were right at the end before)
        // In this case, we only need one copy
        if (rs == 0) {
          assert(current_image_location_in_dataset == 0);
          memcpy(outgoing_msg_send_data_and_ask_grad->content, corpus->labels->physical_get_RCDslice(0), sizeof(float) * corpus->mini_batch_size);
        }
        // Otherwise, we have to copy twice
        else {
          size_t num_images_from_end = corpus->n_images - current_image_location_in_dataset;
          assert(num_images_from_end > 0);
          assert(num_images_from_end < corpus->mini_batch_size);
          size_t num_images_from_beginning = corpus->mini_batch_size - num_images_from_end;
          memcpy(outgoing_msg_send_data_and_ask_grad->content,
            corpus->labels->physical_get_RCDslice(current_image_location_in_dataset),
            sizeof(float) * num_images_from_end);
          memcpy(outgoing_msg_send_data_and_ask_grad->content + num_images_from_end,
            corpus->labels->physical_get_RCDslice(0),
            sizeof(float) * num_images_from_beginning);
        }
        
        // ++current_epoch;
      }
      // Otherwise we will read all of the labels from the corpus
      else {
        // Get labels for this mini batch
        memcpy(outgoing_msg_send_data_and_ask_grad->content, 
          corpus->labels->physical_get_RCDslice(current_image_location_in_dataset),
          sizeof(float) * corpus->mini_batch_size);
      }
      
      // Move forwards in the dataset
      current_image_location_in_dataset += corpus->mini_batch_size;
      if (current_image_location_in_dataset >= corpus->n_images) {
        current_image_location_in_dataset -= corpus->n_images;
      }
      // This assertion isn't needed, it just checks my understanding of how we pass data
      assert(bridges[0]->p_input_layer->p_data_cube->get_p_data() == corpus->images->physical_get_RCDslice(0));
    
      // -----------------------------------------------------------------------
      // Send request to conv model server asking for new model
      // -----------------------------------------------------------------------
      LOG(INFO) << "Sending ASK_MODEL Request..." << std::endl;
      zmq_send (requester, &outgoing_msg_ask_model, outgoing_msg_ask_model.size(), 0);

      // -----------------------------------------------------------------------
      // Receive model
      // -----------------------------------------------------------------------
      LOG(INFO) << "Waiting on ANSWER_MODEL Response..." << std::endl;
      zmq_recv (requester, incoming_model_buf, incoming_model_buf_size, 0);
      OmvMessage * incoming_msg_new_model = reinterpret_cast<OmvMessage*>(incoming_model_buf);
      assert(incoming_msg_new_model->msg_type == ANSWER_MODEL);
      assert(incoming_msg_new_model->size() == outgoing_model_grad_buf_size);
      // Update the network with the model received in this buffer
      DeepNet::set_all_models(bridges, incoming_msg_new_model->content);
      // Debug: Print the first element of the model
      // cout << incoming_msg_new_model->content[0] << endl;
      
      // -----------------------------------------------------------------------
      // Run FW Pass
      // -----------------------------------------------------------------------
      LOG(INFO) << "Running FW Pass..." << std::endl;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        (*bridge)->forward();
      }
      
      // Copy output of last model to outgoing_msg_send_data_and_ask_grad->content
      //
      // Assert that it is on the CPU. For now it should be because we only set
      // the data to be shared with the next bridge in DeepNet using the call to
      // set_share_pointer_with_next_bridge(true). Since this is the last bridge
      // we never called that, i.e. the output will be copied to 
      // p_output_layer->p_data_cube->get_p_data().
      // If we later want to keep this output on the GPU and copy directly from
      // GPU to GPU later, we can set_share_pointer_with_next_bridge here to
      // keep it on the GPU.
      assert(!bridges.back()->get_share_pointer_with_next_bridge());
      // Do a memcpy to outgoing_msg_send_data_and_ask_grad->content
      // This can eventually be a device memcpy (see comment above)
      // Edit: We are offsetting by mini-batch size since we pass labels too
      memcpy(outgoing_msg_send_data_and_ask_grad->content + corpus->mini_batch_size, 
        bridges.back()->p_output_layer->p_data_cube->get_p_data(),
        sizeof(float)*nfloats_output_data);
      // Debug
      // cout << "~" << outgoing_msg_send_data_and_ask_grad->content[0] << std::endl;

      // -----------------------------------------------------------------------
      // Send output of FW Pass to FC Server and simultaneously ask for gradients
      // -----------------------------------------------------------------------
      LOG(INFO) << "Sending output data with Request ASK_GRADIENT_OF_SENT_DATA and waiting for data gradients..." << std::endl;
      // Send data output
      zmq_send (requester_fc, outgoing_msg_send_data_and_ask_grad, outgoing_msg_send_data_and_ask_grad->size(), 0);
      // Wait for data gradients
      LOG(INFO) << "Waiting for ANSWER_GRADIENT_OF_SENT_DATA Response..." << std::endl;
      zmq_recv (requester_fc, incoming_data_grad_buf, incoming_data_grad_buf_size, 0);
      OmvMessage * incoming_msg_data_grads = reinterpret_cast<OmvMessage*>(incoming_data_grad_buf);
      assert(incoming_msg_data_grads->msg_type == ANSWER_GRADIENT_OF_SENT_DATA);
      assert(incoming_msg_data_grads->size() == int(sizeof(OmvMessage) + 1*sizeof(float)*nfloats_output_data));
      LOG(INFO) << "Received data gradients" << std::endl;

      // -----------------------------------------------------------------------
      // Update last layer input data gradients with incoming_msg_data_grads->content
      // -----------------------------------------------------------------------
      assert(!bridges.back()->get_share_pointer_with_next_bridge());
      bridges.back()->update_p_output_layer_gradient_CPU_ONLY(incoming_msg_data_grads->content);
      assert(bridges.back()->p_output_layer->p_gradient_cube->get_p_data() == incoming_msg_data_grads->content);
      // Debug
      // std::cout << incoming_msg_data_grads->content[0] << std::endl;

      // -----------------------------------------------------------------------
      // Run backward loop
      // -----------------------------------------------------------------------
      for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
        (*bridge)->backward();
      }
      // Now that model gradients have all been calculated, fill in outgoing_msg_send_model_grad->content
      DeepNet::get_all_gradients(bridges, outgoing_msg_send_model_grad->content);

      // -----------------------------------------------------------------------
      // Return the gradient to the conv model server
      // -----------------------------------------------------------------------
      LOG(INFO) << "Sending ASK_UPDATE_GRADIENT Request..." << std::endl;
      zmq_send (requester, outgoing_msg_send_model_grad, outgoing_msg_send_model_grad->size(), 0);

      // -----------------------------------------------------------------------
      // Wait until update is done
      // -----------------------------------------------------------------------
      LOG(INFO) << "Waiting for ANSWER_UPDATE_GRADIENT Request..." << std::endl;
      zmq_recv (requester, &incoming_msg_model_grad_updated, incoming_msg_model_grad_updated.size(), 0);
      assert(incoming_msg_model_grad_updated.msg_type == ANSWER_UPDATE_GRADIENT);
      assert(incoming_msg_model_grad_updated.size() == sizeof(OmvMessage));
    }

    // -------------------------------------------------------------------------
    // Destroy network
    // -------------------------------------------------------------------------
    DeepNet::clean_up(bridges, corpus);
    fclose(pFile);

  }
};

#endif

