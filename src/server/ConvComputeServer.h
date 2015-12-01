
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
#include <vector>

#include "sched/TaskQueue.h"

class ConvComputeServer : public Server{
public:

  // SHADJIS TODO: These 3 should be taken from the parent class with using
  std::string name;
  std::string solver_file;
  std::string data_binary;
  
  std::string conv_listen_bind;
  std::string conv_send_bind;
  std::string fc_listen_bind;
  std::string fc_send_bind;

  int group_size;
  int rank_in_group;

  // There are 4 buffers needed:
  // - The model and model gradients (these two are the same size)
  // - The output layer data and output layer gradients (these two are the same size)
  int nfloats_model;
  int nfloats_output_data;

  ConvComputeServer(string _name, std::string _conv_listen_bind, std::string _conv_send_bind,
    std::string _fc_listen_bind, std::string _fc_send_bind, std::string _solver_file, std::string _data_binary,
    int _groupsize, int _rank_in_group) : 
    name(_name), solver_file(_solver_file), data_binary(_data_binary), 
    conv_listen_bind(_conv_listen_bind), conv_send_bind(_conv_send_bind),
    fc_listen_bind(_fc_listen_bind), fc_send_bind(_fc_send_bind),
    group_size(_groupsize), rank_in_group(_rank_in_group),
    nfloats_model(0), nfloats_output_data(0)
  {}

  void start(){
  
    VLOG(2) << "Starting ConvComputeServer[" << name << "]..." << std::endl;

    // -------------------------------------------------------------------------
    // Bind to both servers
    // -------------------------------------------------------------------------
    
    // This server will send 2 messages: sending gradients back to conv server, and sending
    // outputs to fc server. It will also receive a new model (same size as gradients it sends).
     
    int rc;

    void *context_model_agg = zmq_ctx_new ();
    void *responder_model_agg = zmq_socket (context_model_agg, ZMQ_REQ);
    rc = zmq_connect (responder_model_agg, conv_listen_bind.c_str());
    assert (rc == 0);

    void *context_model_broadcast = zmq_ctx_new ();
    void *responder_model_broadcast = zmq_socket (context_model_broadcast, ZMQ_SUB);
    rc = zmq_connect (responder_model_broadcast, conv_send_bind.c_str());
    assert (rc == 0);
    rc = zmq_setsockopt (responder_model_broadcast, ZMQ_SUBSCRIBE, "", 0);
    assert (rc == 0);


    void *context_fc_agg = zmq_ctx_new ();
    void *responder_fc_agg = zmq_socket (context_fc_agg, ZMQ_REQ);
    rc = zmq_connect (responder_fc_agg, fc_listen_bind.c_str());
    assert (rc == 0);

    void *context_fc_broadcast = zmq_ctx_new ();
    void *responder_fc_broadcast = zmq_socket (context_fc_broadcast, ZMQ_SUB);
    rc = zmq_connect (responder_fc_broadcast, fc_send_bind.c_str());
    assert (rc == 0);
    rc = zmq_setsockopt (responder_fc_broadcast, ZMQ_SUBSCRIBE, "", 0);
    assert (rc == 0);

    // -------------------------------------------------------------------------
    // Read parameter files and construct network
    // -------------------------------------------------------------------------
    BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
    // Modify all bridges to not update model gradients in backward pass (saves time)
    for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
      (*bridge)->set_update_model_gradients(false);
    }
    Corpus * const corpus = DeepNet::load_network(solver_file.c_str(), data_binary.c_str(), solver_param, net_param, bridges, true);
    
    // SHADJIS TODO: Later we will call this:
    //   bridges.back()->update_p_output_layer_gradient_CPU_ONLY(&incoming_msg_data_grads->content[rank_in_group*nfloats_output_data]);
    // But the last bridge by default owns the memory of its output cubes, so once we
    // change the pointer it will have a memory leak. We should make a change here to
    // free that cube and replace it with a dummy cube, or specify more intelligently
    // when a layer should not own its output cubes. E.g. we also would not want to if
    // the output of the bridge never needs to get copied back to the host.
    
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
    VLOG(2) << "Allocating " << (1.0*outgoing_model_grad_buf_size/1024/1024) << " MB for outgoing model gradients" << std::endl;
    char * outgoing_model_grad_buf = new char[outgoing_model_grad_buf_size];
    
    // Outgoing data
    // Recall that we need to pass labels as well.
    int outgoing_data_buf_size = sizeof(OmvMessage) + 1*sizeof(float)*(nfloats_output_data + corpus->mini_batch_size);
    VLOG(2) << "Allocating " << (1.0*outgoing_data_buf_size/1024/1024) << " MB for outgoing data" << std::endl;
    char * outgoing_data_buf = new char[outgoing_data_buf_size];

    
    // Outgoing message which sends the model gradients
    OmvMessage * outgoing_msg_send_model_grad = reinterpret_cast<OmvMessage*>(outgoing_model_grad_buf);
    outgoing_msg_send_model_grad->msg_type = ASK_UPDATE_GRADIENT;
    outgoing_msg_send_model_grad->nelem = nfloats_model;
    outgoing_msg_send_model_grad->group_size = group_size;
    outgoing_msg_send_model_grad->rank_in_group = rank_in_group;
    assert(outgoing_msg_send_model_grad->size() == outgoing_model_grad_buf_size);

    // Outgoing message which asks for new model
    // Note: This could have been combined with outgoing_msg_send_model_grad, 
    // i.e. seinding gradients implies asking for new model (like the
    // message below outgoing_msg_send_data_and_ask_grad which does both)
    OmvMessage outgoing_msg_ask_model;
    outgoing_msg_ask_model.msg_type = ASK_MODEL;
    outgoing_msg_ask_model.nelem = 0;
    outgoing_msg_ask_model.group_size = group_size;
    outgoing_msg_ask_model.rank_in_group = rank_in_group;
    assert(outgoing_msg_ask_model.size() == sizeof(OmvMessage));

    // Outgoing message which sends the data and asks for gradients
    OmvMessage * outgoing_msg_send_data_and_ask_grad = reinterpret_cast<OmvMessage*>(outgoing_data_buf);
    outgoing_msg_send_data_and_ask_grad->msg_type = ASK_GRADIENT_OF_SENT_DATA;
    outgoing_msg_send_data_and_ask_grad->nelem = nfloats_output_data + corpus->mini_batch_size;
    outgoing_msg_send_data_and_ask_grad->group_size = group_size;
    outgoing_msg_send_data_and_ask_grad->rank_in_group = rank_in_group;
    assert(outgoing_msg_send_data_and_ask_grad->size() == outgoing_data_buf_size);
    
    // -------------------------------------------
    // Allocate buffer for incoming messages
    // -------------------------------------------
    // Allocate a factor of 2 extra for these although should not be needed
    // The message objects themselves for these 2 will be created inside the loop each iteration
    
    // Incoming model
    int incoming_model_buf_size = sizeof(OmvMessage) + 2*sizeof(float)*nfloats_model;
    VLOG(2) << "Allocating " << (1.0*incoming_model_buf_size/1024/1024) << " MB for incoming model" << std::endl;
    char * incoming_model_buf = new char[incoming_model_buf_size];

    // Incoming data gradients
    int incoming_data_grad_buf_size = sizeof(OmvMessage) + 2*sizeof(float)*nfloats_output_data*group_size;
    VLOG(2) << "Allocating " << (1.0*incoming_data_grad_buf_size/1024/1024) << " MB for incoming data gradients" << std::endl;
    char * incoming_data_grad_buf = new char[incoming_data_grad_buf_size];

    // Incoming message which acknowledges that the model gradients we
    // sent have finished updating, and therefore we can ask for a new model
    // Note: This could have been combined into outgoing_msg_send_model_grad,
    // i.e. sending a model gradient implies waiting for a new model
    OmvMessage incoming_msg_model_grad_updated;
    incoming_msg_model_grad_updated.msg_type = ANSWER_UPDATE_GRADIENT; // This line not needed, in fact should probably not even be set
    incoming_msg_model_grad_updated.nelem = 0;
    incoming_msg_model_grad_updated.group_size = group_size;
    incoming_msg_model_grad_updated.rank_in_group = rank_in_group;
    assert(incoming_msg_model_grad_updated.size() == sizeof(OmvMessage));

    char dummy[50]; // SHADJIS TODO: Not sure what this is, need to add a comment and remove hard-coding of 50

    OmvMessage * incoming_msg_new_model;

    // TASKS
    
    // the first task is to load the data.
    Task task_load_data(
      [&](){
        // -----------------------------------------------------------------------
        // Read in the next mini-batch from file and update labels from lmdb
        // -----------------------------------------------------------------------
        VLOG(2) << "~~~~ ENTER STATE Read corpus" << std::endl;
        
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
      
        VLOG(2) << "~~~~ EXIT STATE Read corpus" << std::endl;
      }
    );

    std::vector<Task> tasks_get_model;
    // for each bridge, there is a task that gets the model.
    int ct1 = 0;
    for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
      tasks_get_model.push_back(
        Task(
          [bridge, ct1, &outgoing_msg_ask_model, &responder_model_agg, &incoming_model_buf,
            &incoming_model_buf_size, &incoming_msg_new_model, &bridges, &dummy,
            &responder_model_broadcast](){

            if((*bridge)->get_model_cube() == NULL){
              VLOG(2) << "------Skipping Bridge " << ct1 << " does not have model" << std::endl;
            }else{
              VLOG(2) << "Sending ASK_MODEL Request Bridge " << ct1 << std::endl;
              VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
              outgoing_msg_ask_model.bridgeid = ct1;
              zmq_send (responder_model_agg, &outgoing_msg_ask_model, outgoing_msg_ask_model.size(), 0);
              zmq_recv (responder_model_agg, dummy, 50, 0);
              VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;

              VLOG(2) << "RECEIVING MODEL FOR BRIDGE " << ct1 << std::endl;
              zmq_recv (responder_model_broadcast, incoming_model_buf, incoming_model_buf_size, 0);
              incoming_msg_new_model = reinterpret_cast<OmvMessage*>(incoming_model_buf);
              VLOG(2) << "RCV MSG " << incoming_msg_new_model->msg_type << "   " << ANSWER_MODEL << std::endl;
              VLOG(2) << "SIZE = " << incoming_msg_new_model->size() << std::endl;
              assert(incoming_msg_new_model->msg_type == ANSWER_MODEL);
              //assert(incoming_msg_new_model->size() == outgoing_model_grad_buf_size);
              DeepNet::set_ith_models(bridges, incoming_msg_new_model->content, ct1);
              VLOG(2) << "RECEIVED MODEL FOR BRIDGE " << ct1 << std::endl;
            }
          }
      ));
      if(ct1 >= 1){
        tasks_get_model[ct1].depend_ons.push_back(tasks_get_model[ct1-1].mymutex);
      }
      ct1 ++;
    }

    std::vector<Task> tasks_forward;
    // for each bridge, there is a task that runs the forward loop.
    int ct = 0;
    for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
      
      tasks_forward.push_back(
        Task(
          [bridge, ct](){
            VLOG(2) << "RUNNING FORWARD FOR BRIDGE " << ct << std::endl;
            assert((*bridge)->get_model_parallelism_group_size() == 1);
            (*bridge)->forward();
            VLOG(2) << "FINISH FORWARD MODEL FOR BRIDGE " << ct << std::endl;
          }
      ));
      if(ct >= 1){
        tasks_forward[ct].depend_ons.push_back(tasks_forward[ct-1].mymutex);
      }
      //tasks_forward[ct].depend_ons.push_back(tasks_get_model[ct].mymutex);
      tasks_forward[ct].depend_ons.push_back(tasks_get_model[ct].mymutex);
      ct ++;
    }
    tasks_forward[0].depend_ons.push_back(task_load_data.mymutex);

    Task task_get_bw_grad(
      [&](){
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
        VLOG(2) << "~~~~ EXIT STATE Copy FW" << std::endl;

        // -----------------------------------------------------------------------
        // Send output of FW Pass to FC Server and simultaneously ask for gradients
        // -----------------------------------------------------------------------
        VLOG(2) << "Sending output data with Request ASK_GRADIENT_OF_SENT_DATA and waiting for data gradients..." << std::endl;
        // Send data output
        VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
        zmq_send (responder_fc_agg, outgoing_msg_send_data_and_ask_grad, outgoing_msg_send_data_and_ask_grad->size(), 0);
        zmq_recv (responder_fc_agg, dummy, 50, 0);
        VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;
        // Wait for data gradients
        VLOG(2) << "Waiting for ANSWER_GRADIENT_OF_SENT_DATA Response..." << std::endl;
        VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
        zmq_recv (responder_fc_broadcast, incoming_data_grad_buf, incoming_data_grad_buf_size, 0);
        VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;
        VLOG(2) << "~~~~ ENTER STATE Read msg" << std::endl;
        OmvMessage * incoming_msg_data_grads = reinterpret_cast<OmvMessage*>(incoming_data_grad_buf);
        assert(incoming_msg_data_grads->msg_type == ANSWER_GRADIENT_OF_SENT_DATA);
        assert(incoming_msg_data_grads->size() == int(sizeof(OmvMessage) + group_size*sizeof(float)*nfloats_output_data));
        VLOG(2) << "~~~~ EXIT STATE Read msg" << std::endl;
        VLOG(2) << "Received data gradients" << std::endl;

        // -----------------------------------------------------------------------
        // Update last layer input data gradients with incoming_msg_data_grads->content
        // -----------------------------------------------------------------------
        VLOG(2) << "~~~~ ENTER STATE Update gradients" << std::endl;
        assert(!bridges.back()->get_share_pointer_with_next_bridge());
        bridges.back()->update_p_output_layer_gradient_CPU_ONLY(&incoming_msg_data_grads->content[rank_in_group*nfloats_output_data]);
        assert(bridges.back()->p_output_layer->p_gradient_cube->get_p_data() == &incoming_msg_data_grads->content[rank_in_group*nfloats_output_data]);
        VLOG(2) << "~~~~ EXIT STATE Update gradients" << std::endl;

        // std::cout << incoming_msg_data_grads->nelem << std::endl;
        // std::cout << rank_in_group << " " << nfloats_output_data << " " << rank_in_group*nfloats_output_data << std::endl;

      }
    );
    task_get_bw_grad.depend_ons.push_back(tasks_forward[ct-1].mymutex);

    std::vector<Task> tasks_backward;
    // for each bridge, there is a task that runs the backward loop.
    int nbridge = ct-1;
    ct = 0;
    for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
      tasks_backward.push_back(
        Task(
          [bridge, ct, nbridge](){
            VLOG(2) << "RUNNING BACKWARD FOR BRIDGE " << (nbridge-ct) << std::endl;
            (*bridge)->backward();
            VLOG(2) << "FINISH BACKWARD MODEL FOR BRIDGE " << (nbridge-ct) << std::endl;
          }
      ));
      if(ct >= 1){
        tasks_backward[ct].depend_ons.push_back(tasks_backward[ct-1].mymutex);
      }
      ct ++;
    }
    tasks_backward[0].depend_ons.push_back(task_get_bw_grad.mymutex);

    std::vector<Task> tasks_update_grad;
    ct = 0;
    for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
      tasks_update_grad.push_back(
        Task(
          [bridge, ct, &bridges, &dummy, &outgoing_msg_send_model_grad, nbridge,
            &responder_model_agg, &responder_model_broadcast, &incoming_msg_model_grad_updated](){
            if((*bridge)->get_model_cube() == NULL){
              VLOG(2) << "------Skipping Bridge (b)" << nbridge-ct << " does not have model" << std::endl;
            }else{
              outgoing_msg_send_model_grad->bridgeid = nbridge-ct;
              size_t model_nelem = DeepNet::get_ith_gradient(bridges, outgoing_msg_send_model_grad->content, nbridge-ct);
              outgoing_msg_send_model_grad->nelem = model_nelem;

              VLOG(2) << "Sending ASK_UPDATE_GRADIENT Request BRIDGE " << nbridge-ct << std::endl;
              VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
              zmq_send (responder_model_agg, outgoing_msg_send_model_grad, outgoing_msg_send_model_grad->size(), 0);
              zmq_recv (responder_model_agg, dummy, 50, 0);
              VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;

              // -----------------------------------------------------------------------
              // Wait until update is done
              // -----------------------------------------------------------------------
              VLOG(2) << "Waiting for ANSWER_UPDATE_GRADIENT Request BRIDGE " << nbridge-ct << std::endl;
              VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
              zmq_recv (responder_model_broadcast, &incoming_msg_model_grad_updated, incoming_msg_model_grad_updated.size(), 0);
              VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;
              assert(incoming_msg_model_grad_updated.msg_type == ANSWER_UPDATE_GRADIENT);
              assert(incoming_msg_model_grad_updated.size() == sizeof(OmvMessage));
            }
          }
      ));
      if(ct >= 1){
        tasks_update_grad[ct].depend_ons.push_back(tasks_update_grad[ct-1].mymutex);
      }
      tasks_update_grad[ct].depend_ons.push_back(tasks_backward[ct].mymutex);
      ct ++;
    }

    // Create Queuq
    TaskQueue queue;
    queue.tasks.push_back(task_load_data);
    for(size_t i=0;i<tasks_get_model.size();i++){
      queue.tasks.push_back(tasks_get_model[i]);
    }
    for(size_t i=0;i<tasks_forward.size();i++){
      queue.tasks.push_back(tasks_forward[i]);
    }
    queue.tasks.push_back(task_get_bw_grad);
    for(size_t i=0;i<tasks_backward.size();i++){
      queue.tasks.push_back(tasks_backward[i]);
    }
    for(size_t i=0;i<tasks_update_grad.size();i++){
      queue.tasks.push_back(tasks_update_grad[i]);
    }

    // -------------------------------------------------------------------------
    // Main Loop
    // -------------------------------------------------------------------------
    while(1){
    
      queue.prepare();
      queue.run();

    }

    // -------------------------------------------------------------------------
    // Destroy network
    // -------------------------------------------------------------------------
    DeepNet::clean_up(bridges, corpus);
    fclose(pFile);

  }
};

#endif

