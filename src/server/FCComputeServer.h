
#ifndef _FCCOMPUTESERVER_H
#define _FCCOMPUTESERVER_H

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

class FCComputeServer : public Server{
public:

  // SHADJIS TODO: These 3 should be taken from the parent class with using
  std::string name;
  std::string solver_file;
  
  // The Conv Compute server needs to listen to conv model server and fc compute server
  // (or the special-case fc compute + model server)
  // The fc compute server is similar, but reversed: it listens to the fc model
  // server and the conv compute server
  std::string fc_listen_bind;
  std::string fc_send_bind;
  std::string conv_listen_bind;
  std::string conv_send_bind;

  int group_size;       // Incoming (from CC)
  int rank_in_group;

  // There are 4 buffers needed:
  // - The model and model gradients (these two are the same size)
  // - The output layer data and output layer gradients (these two are the same size)
  int nfloats_model;
  int nfloats_data;

  FCComputeServer(string _name, std::string _fc_listen_bind, std::string _fc_send_bind,
    std::string _conv_listen_bind, std::string _conv_send_bind, std::string _solver_file,
    int _groupsize, int _rank_in_group) : 
    name(_name), solver_file(_solver_file),
    fc_listen_bind(_fc_listen_bind), fc_send_bind(_fc_send_bind),
    conv_listen_bind(_conv_listen_bind), conv_send_bind(_conv_send_bind),
    group_size(_groupsize), rank_in_group(_rank_in_group),
    nfloats_model(0), nfloats_data(0)
  {}

  // ------------------------------------------------------------------------------------------------
  // Summmary of tasks
  // ------------------------------------------------------------------------------------------------
  // The conv compute server is defined by 6 types of tasks (see comment in that class)
  // The fc compute server is defined by tasks to:
  //  - Read data for fw pass (from conv compute server)
  //  - Run each fw pass (these tasks are per-bridge)
  //  - Ask for each model (these tasks are per-bridge)
  //  - Run each bw pass (these tasks are per-bridge)
  //  - Send back each model gradient (these tasks are per-bridge)
  //  - Print out loss and send data gradients back to each CC (using broker)
  //
  // The dependncies of these tasks are straightforward except for 2 things:
  //  - Reading the data for fw pass involves a broker / UDF (should not be too complicated)
  //  - Running fw/bw is usually 1 task per bridge, but in the case of model parallelism
  //    all parallel bridges needed to be run as part of that task, and then the model update
  //    and gradient update needs to be reformatted since the fc model server expects a different format
  //
  // Unlike CC which has a while(1) loop and runs the task queue repeatedly, here we have a broker
  // to get data from all CC servers and then runs a UDF. The UDF replaces the while(1) loop and each
  // time the UDF runs it runs the queue.
  //
  // SHADJIS TODO: The model is large -- can this ever really help to have 2 FC servers?
  // It might no longer be beneficial to have model parallelism and fc compute servers, i.e.
  // once we have to read / concat from each it might be faster to just have 1 GPU?
  //
  // Also, eventually we need to support more than 1 cc group per fc
  // (and similarly for CC, more than 1 batch per CC)
  //
  // Note that currently fccm (fc compute/model server) has a lock, but fcc does not (like cc does not),
  // since the current fcc implementation is one fc per group. Then fcm has the lock. If we supported
  // more than 1 cc group per fcc, we would need a lock now (i.e. thread/lambdas point below)
  //
  // ------------------------------------------------------------------------------------------------
  // Summary of broker (within group), multiple threads/ports (across groups), task queue (hide latency/async)
  // ------------------------------------------------------------------------------------------------
  // So right now we assert each fcc has 1 cc group, i.e. need only 1 broker
  // But if we want to support more, then we need the parallel threads (like in cm server). Specifically:
  //    - if you listen to multiple groups (like cm and fcm do), need threads and lambda for each
  //    - if you listen to a group of size > 1, like cm and fcm do, need a broker (and therefore UDF)
  //    - if you want async (hide latency of model/grad updates and computation), need task queue (like cc does)
  // Note FCC is unique because it both has async to hide model/grad updates (like cc) but also because it
  // has to aggregate all the cc in a group. So it needs both tasks as well as a broker/UDF, i.e. the entire
  // task queue will be inside a udf which runs the queue when the broker has all the data ready.
  // To then support more than 1 cc group per fcc, 
  // we need multiple threads (one per listening group), each with a broker/udf, each having its task queue.
  // The reason we would have more than 1 group is to hide latency: while the cc is computing, fcc is doing nothing
  // (it could be handling another cc group).
  // Similarly, we can hide more latency in cc by having each cc run multiple groups at once (idle while fcc/fccm runs)
  // 
  // ------------------------------------------------------------------------------------------------
  // Summary of hiding latency:
  // ------------------------------------------------------------------------------------------------
  // But note we can hide more latency:
  //    - cc now hides latency of sending models with its computation, but while fc runs it is idle
  //        - fix: have multiple batches per cc and run it while fc is idle
  //        - this requires multiple task queues
  //    - fccm can hide latency of sending data/grad while doing compute (sending is fast but so is compute so matters)
  //        - no issue of waiting for fc to finish (like there is for cc/fcc) since we can just add more fc then
  //        - fix: make multiple tasks?
  //    - fcc can also hide latency of sending models (like cc does) and also it is idle while waiting for cc to
  //      finish (so can wait on multiple groups as described above using threads + queues + broker).
  //    - and fcc and cc may also be able to hide latency of sending data, like 2nd point above of fccm
  //
  
  
  void start(){
  
    VLOG(2) << "Starting FCComputeServer[" << name << "]..." << std::endl;

    // -------------------------------------------------------------------------
    // Bind to fc model server (fc compute server binding handled by broker)
    // -------------------------------------------------------------------------
    
    // This server will send 2 messages: sending model gradients back to fc server, and sending data
    // gradients to conv server. It will also receive a model (same size as model gradients it sends),
    // and receive data from the conv server.
     
    int rc;

    void *context_model_agg = zmq_ctx_new ();
    void *responder_model_agg = zmq_socket (context_model_agg, ZMQ_REQ);
    rc = zmq_connect (responder_model_agg, fc_listen_bind.c_str());
    assert (rc == 0);

    void *context_model_broadcast = zmq_ctx_new ();
    void *responder_model_broadcast = zmq_socket (context_model_broadcast, ZMQ_SUB);
    rc = zmq_connect (responder_model_broadcast, fc_send_bind.c_str());
    assert (rc == 0);
    rc = zmq_setsockopt (responder_model_broadcast, ZMQ_SUBSCRIBE, "", 0);
    assert (rc == 0);

    // -------------------------------------------------------------------------
    // Read parameter files and construct network
    // -------------------------------------------------------------------------
    BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
    Corpus * const corpus = DeepNet::load_network(solver_file.c_str(), solver_param, net_param, bridges, true);
    // Modify all bridges to not update model gradients in backward pass (saves time)
    for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
      (*bridge)->set_update_model_gradients(false);
    }
    SoftmaxBridge * const softmax = (SoftmaxBridge *) bridges.back();
    LogicalCubeFloat * const labels = softmax->p_data_labels;
    
    // For printing information
    const int display_iter = solver_param.display();
    const int snapshot = solver_param.snapshot();
    float loss = 0.;
    float accuracy = 0.;
    int batch = 0;
    Timer timer;
    
    // -------------------------------------------------------------------------
    // Allocate buffers and create messages
    // -------------------------------------------------------------------------
    
    // Size of model
    nfloats_model = DeepNet::get_parameter_size(bridges);
    // Size of the first layer
    nfloats_data  = bridges.front()->get_input_data_size();

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
    
    // Outgoing data gradients
    // Recall that we do not need to pass labels in backwards pass.
    int outgoing_data_grad_buf_size = sizeof(OmvMessage) + 1*sizeof(float)*(nfloats_data);
    VLOG(2) << "Allocating " << (1.0*outgoing_data_grad_buf_size/1024/1024) << " MB for outgoing data" << std::endl;
    char * outgoing_data_grad_buf = new char[outgoing_data_grad_buf_size];

    
    // Outgoing message which sends the model gradients
    OmvMessage * outgoing_msg_send_model_grad = reinterpret_cast<OmvMessage*>(outgoing_model_grad_buf);
    outgoing_msg_send_model_grad->msg_type = ASK_UPDATE_GRADIENT;
    outgoing_msg_send_model_grad->nelem = nfloats_model;
    outgoing_msg_send_model_grad->group_size = 1;               // This is group size sent to FC Model server. Group size from CC server is different.
    outgoing_msg_send_model_grad->rank_in_group = rank_in_group;
    assert(outgoing_msg_send_model_grad->size() == outgoing_model_grad_buf_size);

    // Outgoing message which asks for new model
    // Note: This could have been combined with outgoing_msg_send_model_grad, 
    // i.e. seinding gradients implies asking for new model (like the
    // CC message outgoing_msg_send_data_and_ask_grad which does both)
    // But it is better for statistical efficiency to wait until that model is needed
    // (as late as possible) and ask for it during the next fw pass instead
    OmvMessage outgoing_msg_ask_model;
    outgoing_msg_ask_model.msg_type = ASK_MODEL;
    outgoing_msg_ask_model.nelem = 0;
    outgoing_msg_ask_model.group_size = 1;
    outgoing_msg_ask_model.rank_in_group = rank_in_group;
    assert(outgoing_msg_ask_model.size() == sizeof(OmvMessage));

    // Create the response message for returning the data gradients
    OmvMessage * outgoing_msg_data_grad = reinterpret_cast<OmvMessage*>(outgoing_data_grad_buf);
    outgoing_msg_data_grad->msg_type = ANSWER_GRADIENT_OF_SENT_DATA;
    outgoing_msg_data_grad->nelem = nfloats_data;
    assert(outgoing_msg_data_grad->size() == outgoing_data_grad_buf_size);
    
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
    int incoming_data_buf_size = sizeof(OmvMessage) + 2*sizeof(float)*(nfloats_data + corpus->mini_batch_size);
    VLOG(2) << "Allocating " << (1.0*incoming_data_buf_size/1024/1024) << " MB for incoming data gradients" << std::endl;
    char * incoming_data_buf = new char[incoming_data_buf_size];

    // Incoming message which acknowledges that the model gradients we
    // sent have finished updating, and therefore we can ask for a new model
    // Note: This could have been combined into outgoing_msg_send_model_grad,
    // i.e. sending a model gradient implies waiting for a new model
    OmvMessage incoming_msg_model_grad_updated;
    incoming_msg_model_grad_updated.msg_type = ANSWER_UPDATE_GRADIENT; // This line not needed, in fact should probably not even be set
    incoming_msg_model_grad_updated.nelem = 0;
    incoming_msg_model_grad_updated.group_size = 1;
    incoming_msg_model_grad_updated.rank_in_group = rank_in_group;
    assert(incoming_msg_model_grad_updated.size() == sizeof(OmvMessage));

    char dummy[50]; // SHADJIS TODO: Not sure what this is, need to add a comment and remove hard-coding of 50

    OmvMessage * incoming_msg_new_model;
    
    // UDF
    std::vector<OmvMessage *> tmp_msgs;
    tmp_msgs.resize(group_size);
    int tmp_nmsg;

    // =============================================================================================
    // TASKS
    // =============================================================================================
    // Tasks are necessary to hide latency (async model/gradient updates) 
    // Right now they use dummy pointers, these will be reassigned by UDF
    
    // ---------------------------------------------------------------------------------------------
    // TASK -- Create a broker to get the data from conv compute
    // ---------------------------------------------------------------------------------------------
    Task task_get_fw_data(
      [&](){
        VLOG(2) << "~~~~ ENTER STATE Read data from conv compute" << std::endl;
        
        OmvMessage * incoming_msg_data = reinterpret_cast<OmvMessage *>(incoming_data_buf);
        const int nlabel_per_msg = corpus->mini_batch_size/tmp_nmsg;
        const int ndata_per_msg  = tmp_msgs[0]->nelem - nlabel_per_msg;

        // Init incoming_msg_data from different inputs.
        incoming_msg_data->msg_type = tmp_msgs[0]->msg_type;
        incoming_msg_data->nelem = (ndata_per_msg+nlabel_per_msg)*tmp_nmsg;

        // Put together the incoming data in the right order
        for(int i=0;i<tmp_nmsg;i++){
          memcpy(&incoming_msg_data->content[i*nlabel_per_msg], tmp_msgs[i]->content, sizeof(float)*nlabel_per_msg);
          memcpy(&incoming_msg_data->content[corpus->mini_batch_size + i*ndata_per_msg], 
                  tmp_msgs[i]->content + nlabel_per_msg, 
                  sizeof(float)*ndata_per_msg);
        }

        assert(incoming_msg_data->msg_type == ASK_GRADIENT_OF_SENT_DATA);
        assert(incoming_msg_data->size() == int(sizeof(OmvMessage) + 1*sizeof(float)*(nfloats_data + corpus->mini_batch_size)));

        // Update input layer to point to the incoming batch of data
        VLOG(2) << "~~~~ ENTER STATE Update input layer" << std::endl;
        bridges[0]->update_p_input_layer_data_CPU_ONLY(incoming_msg_data->content + corpus->mini_batch_size);
        assert(bridges[0]->p_input_layer->p_data_cube->get_p_data() == incoming_msg_data->content + corpus->mini_batch_size);
        // Initialize labels for this mini batch
        labels->set_p_data(incoming_msg_data->content);
        VLOG(2) << "~~~~ EXIT STATE Update input layer" << std::endl;
        
        // Also reset softmax loss
        softmax->reset_loss();
          
        VLOG(2) << "~~~~ EXIT STATE Read data from conv compute" << std::endl;
      }
    );

    // ---------------------------------------------------------------------------------------------
    // TASK -- Get models from fc model server (one task per bridge)
    // ---------------------------------------------------------------------------------------------
    std::vector<Task> tasks_get_model;
    // SHADJIS TODO: num_prev_tasks redundant, can determine from tasks_get_model.size()
    int num_prev_tasks = 0;
    // SHADJIS TODO: These two are hacks to handle model parallelism
    int bridge_id_no_model_p = 0;
    int bridge_id_with_model_p = 0;
    for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
    
      // We need to push back a task to get the model for this bridge
      // Unfortunately, because of how model parallelism is implemented,
      // this may be multiple bridges. So check how many bridges there are:
      const int num_bridges_in_this_group = (*bridge)->get_model_parallelism_group_size();
      
      // If just one bridge, then set the model normally in this task
      if (num_bridges_in_this_group == 1) {
        tasks_get_model.push_back(
          Task(
            [bridge, bridge_id_no_model_p, bridge_id_with_model_p, &outgoing_msg_ask_model, &responder_model_agg, &incoming_model_buf,
              &incoming_model_buf_size, &incoming_msg_new_model, &bridges, &dummy,
              &responder_model_broadcast](){
        
              if((*bridge)->get_model_cube() == NULL){
                VLOG(2) << "------Skipping Bridge " << (*bridge)->name << " does not have model" << std::endl;
              }else{
                VLOG(2) << "Sending ASK_MODEL Request Bridge " << bridge_id_no_model_p << std::endl;
                VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
                outgoing_msg_ask_model.bridgeid = bridge_id_no_model_p;
                zmq_send (responder_model_agg, &outgoing_msg_ask_model, outgoing_msg_ask_model.size(), 0);
                zmq_recv (responder_model_agg, dummy, 50, 0);
                VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;
        
                VLOG(2) << "RECEIVING MODEL FOR BRIDGE " << bridge_id_no_model_p << std::endl;
                zmq_recv (responder_model_broadcast, incoming_model_buf, incoming_model_buf_size, 0);
                incoming_msg_new_model = reinterpret_cast<OmvMessage*>(incoming_model_buf);
                VLOG(2) << "RCV MSG " << incoming_msg_new_model->msg_type << "   " << ANSWER_MODEL << std::endl;
                VLOG(2) << "SIZE = " << incoming_msg_new_model->size() << std::endl;
                assert(incoming_msg_new_model->msg_type == ANSWER_MODEL);
                //assert(incoming_msg_new_model->size() == outgoing_model_grad_buf_size);
                DeepNet::set_ith_models(bridges, incoming_msg_new_model->content, bridge_id_with_model_p);
                VLOG(2) << "RECEIVED MODEL FOR BRIDGE " << bridge_id_no_model_p << std::endl;
              }
            }
        ));
      }
      // Otherwise this is a bridge with model parallelism, so launch threads
      else {
        bridge_id_no_model_p -= 1; // To account for split
        tasks_get_model.push_back(
          Task(
            [bridge, bridge_id_no_model_p, bridge_id_with_model_p, num_bridges_in_this_group, &outgoing_msg_ask_model, &responder_model_agg, &incoming_model_buf,
              &incoming_model_buf_size, &incoming_msg_new_model, &bridges, &dummy,
              &responder_model_broadcast](){
        
              assert((*bridge)->get_model_cube());
              VLOG(2) << "Sending ASK_MODEL Request Bridge " << (*bridge)->name << std::endl;
              VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
              outgoing_msg_ask_model.bridgeid = bridge_id_no_model_p;
              zmq_send (responder_model_agg, &outgoing_msg_ask_model, outgoing_msg_ask_model.size(), 0);
              zmq_recv (responder_model_agg, dummy, 50, 0);
              VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;
        
              VLOG(2) << "RECEIVING MODEL FOR BRIDGE " << bridge_id_no_model_p << std::endl;
              zmq_recv (responder_model_broadcast, incoming_model_buf, incoming_model_buf_size, 0);
              incoming_msg_new_model = reinterpret_cast<OmvMessage*>(incoming_model_buf);
              VLOG(2) << "RCV MSG " << incoming_msg_new_model->msg_type << "   " << ANSWER_MODEL << std::endl;
              VLOG(2) << "SIZE = " << incoming_msg_new_model->size() << std::endl;
              assert(incoming_msg_new_model->msg_type == ANSWER_MODEL);
              VLOG(2) << "RECEIVED MODEL FOR BRIDGE " << bridge_id_no_model_p << std::endl;
              
              // Now is where we differ from above. In the normal case, we would just call:
              //    DeepNet::set_ith_models(bridges, incoming_msg_new_model->content, bridge_id_no_model_p);
              // However now we want to set multiple models, and we also need to reformat the data
              // This is because the model server is always on the CPU (for now) so the model is stored
              // contiguously. But because model parallelism is implemented as separate bridges, we
              // need to reformat the model. Summary:
              //
              //    incoming_model_buf:       (model, bias)
              //        - AKA incoming_msg_new_model->content
              //    format we need:           (model_part1, bias_part1), (model_part2, bias_part2), ...
              //
              
              // 1. Get the total sizes of the models
              std::vector<int> model_sizes;
              std::vector<int> bias_sizes;
              int total_model_size = 0;
              auto bridge_mutable = bridge;
              for (int b = 0; b < num_bridges_in_this_group; ++b) {
                model_sizes.push_back ((*bridge_mutable)->get_model_cube()->n_elements);
                total_model_size +=    (*bridge_mutable)->get_model_cube()->n_elements;
                bias_sizes .push_back ((*bridge_mutable)->get_bias_cube ()->n_elements);
                ++bridge_mutable; // Update the copy
              }
              // 2. Copy from incoming_msg_new_model->content to each bridge
              int model_offset = 0;
              int bias_offset = 0;
              for (int b = 0; b < num_bridges_in_this_group; ++b) {
                // Copy model and bias
                DeepNet::set_ith_model_only(bridges, incoming_msg_new_model->content + model_offset, bridge_id_with_model_p + b);
                DeepNet::set_ith_bias_only (bridges, incoming_msg_new_model->content + total_model_size + bias_offset, bridge_id_with_model_p + b);
                model_offset += model_sizes[b];
                bias_offset  += bias_sizes[b];
              }
            }
        ));
        // Now move ahead the bridge iterator
        // SHADJIS TODO: find a better way to skip this, e.g. like in DeepNet::run_forward_pass
        for (int b = 0; b < num_bridges_in_this_group-1; ++b) {
          ++bridge; // This is ok since it is not a copy (not inside a lambda)
        }
        bridge_id_with_model_p += num_bridges_in_this_group - 1; // -1 since we incr every iter anyway
        bridge_id_no_model_p -= 1; // To account for funnel
      }
      // Make sure each new task depends on the previous one
      if(num_prev_tasks >= 1){
        tasks_get_model[num_prev_tasks].dependencies.push_back(tasks_get_model[num_prev_tasks-1].mymutex);
      }
      ++num_prev_tasks;
      ++bridge_id_no_model_p;
      ++bridge_id_with_model_p;
    }

    // ---------------------------------------------------------------------------------------------
    // TASK -- Run FW pass (one task per bridge)
    // ---------------------------------------------------------------------------------------------
    std::vector<Task> tasks_forward;
    // SHADJIS TODO: num_prev_tasks redundant, can determine from tasks_get_model.size()
    num_prev_tasks = 0;
    for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
      
      // We need to push back a task to run the fw pass on this bridge
      // Unfortunately, because of how model parallelism is implemented,
      // this may be multiple bridges. So check how many bridges there are:
      const int num_bridges_in_this_group = (*bridge)->get_model_parallelism_group_size();
      
      // If just one bridge, then run forward normally in this task
      if (num_bridges_in_this_group == 1) {
        tasks_forward.push_back(
          Task(
            [bridge](){
              VLOG(2) << "RUNNING FORWARD FOR BRIDGE " << (*bridge)->name << std::endl;
              (*bridge)->forward();
              VLOG(2) << "FINISH FORWARD FOR BRIDGE " << (*bridge)->name << std::endl;
            }
        ));
      }
      // Otherwise this is a bridge with model parallelism, so launch threads
      // SHADJIS TODO: Currently I am doing this with a single task and 4 threads inside it
      // Another way to do it would be with 4 tasks that all have the same depenencies
      // and which can therefore run in parallel. Fix during refactoring.
      // (Note: In the case where we will handle many groups in fcc servers, this will be
      // a thread (for the model parallelism) inside a task (which is async) inside a thread..)
      else {
        tasks_forward.push_back(
          Task(
            [num_bridges_in_this_group, bridge](){
              std::vector<std::thread> threads;
              auto bridge_mutable = bridge;
              for (int b = 0; b < num_bridges_in_this_group; ++b) {
                assert((*bridge_mutable)->get_model_parallelism_group_size() == num_bridges_in_this_group);
                threads.push_back(thread([bridge_mutable]() {
                  VLOG(2) << "RUNNING FORWARD FOR MODEL PARALLEL BRIDGE " << (*bridge_mutable)->name << std::endl;
                  (*bridge_mutable)->forward();
                  VLOG(2) << "FINISH FORWARD FOR MODEL PARALLEL BRIDGE " << (*bridge_mutable)->name << std::endl;
                }));
                // SHADJIS TODO: find a better way to skip this
                ++bridge_mutable; // This will change the copy (not the original outside the lambda)
              }
              // Join
              for (size_t ti = 0; ti < threads.size(); ti++) {
                threads[ti].join();
              }
            }
        ));
        // Now move ahead the bridge iterator
        // SHADJIS TODO: find a better way to skip this, e.g. like in DeepNet::run_forward_pass
        for (int b = 0; b < num_bridges_in_this_group-1; ++b) {
          ++bridge; // This is ok since it is not a copy (not inside a lambda)
        }
      }
      // Make sure each new task depends on the previous one
      if(num_prev_tasks >= 1){
        tasks_forward[num_prev_tasks].dependencies.push_back(tasks_forward[num_prev_tasks-1].mymutex);
      }
      //tasks_forward[num_prev_tasks].dependencies.push_back(tasks_get_model[num_prev_tasks].mymutex);
      tasks_forward[num_prev_tasks].dependencies.push_back(tasks_get_model[num_prev_tasks].mymutex);
      ++num_prev_tasks;
    }
    tasks_forward[0].dependencies.push_back(task_get_fw_data.mymutex);

    // ---------------------------------------------------------------------------------------------
    // TASK -- Once FW pass is done for all bridges, calculate accuracy
    // ---------------------------------------------------------------------------------------------
    // I combined this into UDF, can delete now
    //
    // Task task_update_acc(
    //   [&](){
    //     VLOG(2) << "~~~~ ENTER STATE ACC" << std::endl;
    //     loss += (softmax->get_loss() / float(corpus->mini_batch_size));
    //     accuracy += float(DeepNet::find_accuracy(labels, (*--bridges.end())->p_output_layer->p_data_cube)) / float(corpus->mini_batch_size);
    //     VLOG(2) << "~~~~ EXIT STATE ACC" << std::endl;
    //   }
    // );
    // task_update_acc.dependencies.push_back(tasks_forward[num_prev_tasks-1].mymutex);

    // ---------------------------------------------------------------------------------------------
    // TASK -- Run BW pass (one task per bridge)
    // ---------------------------------------------------------------------------------------------
    std::vector<Task> tasks_backward;
    // SHADJIS TODO: num_prev_tasks redundant, can determine from tasks_get_model.size()
    num_prev_tasks = 0;
    for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
    
      // As in fw pass, run model parallel bridges using threads
      const int num_bridges_in_this_group = (*bridge)->get_model_parallelism_group_size();
      
      // If just one bridge, then run forward normally in this task
      if (num_bridges_in_this_group == 1) {
        tasks_backward.push_back(
          Task(
            [bridge](){
              VLOG(2) << "RUNNING BACKWARD FOR BRIDGE " << (*bridge)->name << std::endl;
              (*bridge)->backward();
              VLOG(2) << "FINISH BACKWARD FOR BRIDGE " << (*bridge)->name << std::endl;
            }
        ));
      }
      // Otherwise this is a bridge with model parallelism, so launch threads
      // SHADJIS TODO: Currently I am doing this with a single task and 4 threads inside it
      // Another way to do it would be with 4 tasks that all have the same depenencies
      // and which can therefore run in parallel. Fix during refactoring. 
      else {
        tasks_backward.push_back(
          Task(
            [num_bridges_in_this_group, bridge](){
              std::vector<std::thread> threads;
              auto bridge_mutable = bridge;
              for (int b = 0; b < num_bridges_in_this_group; ++b) {
                assert((*bridge_mutable)->get_model_parallelism_group_size() == num_bridges_in_this_group);
                threads.push_back(thread([bridge_mutable](){
                  VLOG(2) << "RUNNING BACKWARD FOR MODEL PARALLEL BRIDGE " << (*bridge_mutable)->name << std::endl;
                  (*bridge_mutable)->backward();
                  VLOG(2) << "FINISH BACKWARD FOR MODEL PARALLEL BRIDGE " << (*bridge_mutable)->name << std::endl;
                }));
                // SHADJIS TODO: find a better way to skip this
                ++bridge_mutable; // This will change the copy (not the original outside the lambda)
              }
              // Join
              for (size_t ti = 0; ti < threads.size(); ti++) {
                threads[ti].join();
              }
            }
        ));
        // Now move ahead the bridge iterator
        // SHADJIS TODO: find a better way to skip this, e.g. like in DeepNet::run_forward_pass
        for (int b = 0; b < num_bridges_in_this_group-1; ++b) {
          ++bridge; // This is ok since it is not a copy (not inside a lambda)
        }
      }
      if(num_prev_tasks >= 1){
        tasks_backward[num_prev_tasks].dependencies.push_back(tasks_backward[num_prev_tasks-1].mymutex);
      }
      num_prev_tasks ++;
    }
    tasks_backward[0].dependencies.push_back(tasks_forward.back().mymutex);

    // ---------------------------------------------------------------------------------------------
    // TASK -- Send gradients to fc model server (one task per bridge)
    // ---------------------------------------------------------------------------------------------
    std::vector<Task> tasks_update_grad;
    // SHADJIS TODO: num_prev_tasks redundant, can determine from tasks_get_model.size()
    num_prev_tasks = 0;
    // SHADJIS TODO: These two are hacks to handle model parallelism
    // We will re-use them from before, since they previously (when sending FW models) were at final bridge id + 1
    --bridge_id_no_model_p;
    --bridge_id_with_model_p;
    for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
    
      // We need to push back a task to get the model for this bridge
      // Unfortunately, because of how model parallelism is implemented,
      // this may be multiple bridges. So check how many bridges there are:
      const int num_bridges_in_this_group = (*bridge)->get_model_parallelism_group_size();
      
      // If just one bridge, then set the model normally in this task
      if (num_bridges_in_this_group == 1) {
        tasks_update_grad.push_back(
          Task(
            [bridge, bridge_id_no_model_p, bridge_id_with_model_p, &bridges, &dummy, &outgoing_msg_send_model_grad,
              &responder_model_agg, &responder_model_broadcast, &incoming_msg_model_grad_updated](){
              if((*bridge)->get_model_cube() == NULL){
                VLOG(2) << "------Skipping Bridge " << (*bridge)->name << " does not have model" << std::endl;
              }else{
                outgoing_msg_send_model_grad->bridgeid = bridge_id_no_model_p;
                size_t model_nelem = DeepNet::get_ith_gradient(bridges, outgoing_msg_send_model_grad->content, bridge_id_with_model_p);
                outgoing_msg_send_model_grad->nelem = model_nelem;
        
                VLOG(2) << "Sending ASK_UPDATE_GRADIENT Request BRIDGE " << bridge_id_no_model_p << std::endl;
                VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
                zmq_send (responder_model_agg, outgoing_msg_send_model_grad, outgoing_msg_send_model_grad->size(), 0);
                zmq_recv (responder_model_agg, dummy, 50, 0);
                VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;
        
                // -----------------------------------------------------------------------
                // Wait until update is done
                // -----------------------------------------------------------------------
                VLOG(2) << "Waiting for ANSWER_UPDATE_GRADIENT Request BRIDGE " << bridge_id_no_model_p << std::endl;
                VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
                zmq_recv (responder_model_broadcast, &incoming_msg_model_grad_updated, incoming_msg_model_grad_updated.size(), 0);
                VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;
                assert(incoming_msg_model_grad_updated.msg_type == ANSWER_UPDATE_GRADIENT);
                assert(incoming_msg_model_grad_updated.size() == sizeof(OmvMessage));
              }
            }
        ));
      }
      // Otherwise this is a bridge with model parallelism, so launch threads
      else {
        bridge_id_no_model_p += 1; // To account for funnel
        tasks_update_grad.push_back(
          Task(
            [bridge, bridge_id_no_model_p, bridge_id_with_model_p, num_bridges_in_this_group, &bridges, &dummy, &outgoing_msg_send_model_grad,
              &responder_model_agg, &responder_model_broadcast, &incoming_msg_model_grad_updated](){
              
              assert((*bridge)->get_model_cube());
              
              // Unlike above, we have to reformat the gradients first
              // See comments in the get model tasks for more information
        
              // 1. Get the total sizes of the models
              std::vector<int> model_sizes;
              std::vector<int> bias_sizes;
              int total_model_size = 0;
              int total_model_and_bias_size = 0;
              auto bridge_mutable = bridge;
              // Note that unlike the fw pass get model tasks, here we traverse in reverse
              for (int b = 0; b < num_bridges_in_this_group; ++b) {
                model_sizes.push_back ((*bridge_mutable)->get_model_cube()->n_elements);
                total_model_size    += (*bridge_mutable)->get_model_cube()->n_elements;
                bias_sizes .push_back ((*bridge_mutable)->get_bias_cube ()->n_elements);
                total_model_and_bias_size += (*bridge_mutable)->get_model_cube()->n_elements;
                total_model_and_bias_size += (*bridge_mutable)->get_bias_cube()->n_elements;
                ++bridge_mutable; // Update the copy
              }
              outgoing_msg_send_model_grad->bridgeid = bridge_id_no_model_p;
              outgoing_msg_send_model_grad->nelem = total_model_and_bias_size;
              
              // 2. Copy from each bridge to outgoing_msg_send_model_grad->content
              int model_offset = 0;
              int bias_offset = 0;
              for (int b = 0; b < num_bridges_in_this_group; ++b) {
                // We want to fill the gradient buffer in the order bridge0 bridge1 bridge2 bridge3 etc.,
                // but note we are iterating in reverse, so convert 3 to 0, 2 to 1, etc.:
                const int b_idx = num_bridges_in_this_group - 1 - b;
                // Copy model and bias
                DeepNet::get_ith_gradient_model_only(bridges, outgoing_msg_send_model_grad->content + model_offset, bridge_id_with_model_p - b_idx);
                DeepNet::get_ith_gradient_bias_only (bridges, outgoing_msg_send_model_grad->content + total_model_size + bias_offset, bridge_id_with_model_p - b_idx);
                model_offset += model_sizes[b_idx];
                bias_offset  += bias_sizes [b_idx];
              }
              
              // Now send the model gradients
              VLOG(2) << "Sending ASK_UPDATE_GRADIENT Request BRIDGE " << bridge_id_no_model_p << std::endl;
              VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
              zmq_send (responder_model_agg, outgoing_msg_send_model_grad, outgoing_msg_send_model_grad->size(), 0);
              zmq_recv (responder_model_agg, dummy, 50, 0);
              VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;
        
              // -----------------------------------------------------------------------
              // Wait until update is done
              // -----------------------------------------------------------------------
              VLOG(2) << "Waiting for ANSWER_UPDATE_GRADIENT Request BRIDGE " << bridge_id_no_model_p << std::endl;
              VLOG(2) << "~~~~ ENTER STATE IDLE" << std::endl;
              zmq_recv (responder_model_broadcast, &incoming_msg_model_grad_updated, incoming_msg_model_grad_updated.size(), 0);
              VLOG(2) << "~~~~ EXIT STATE IDLE" << std::endl;
              assert(incoming_msg_model_grad_updated.msg_type == ANSWER_UPDATE_GRADIENT);
              assert(incoming_msg_model_grad_updated.size() == sizeof(OmvMessage));
              
            }
        ));
        // Now move ahead the bridge iterator
        // SHADJIS TODO: find a better way to skip this, e.g. like in DeepNet::run_forward_pass
        for (int b = 0; b < num_bridges_in_this_group-1; ++b) {
          ++bridge; // This is ok since it is not a copy (not inside a lambda)
        }
        bridge_id_with_model_p -= num_bridges_in_this_group - 1; // -1 since we incr every iter anyway
        bridge_id_no_model_p += 1; // To account for split
      }
      if(num_prev_tasks >= 1){
        tasks_update_grad[num_prev_tasks].dependencies.push_back(tasks_update_grad[num_prev_tasks-1].mymutex);
      }
      tasks_update_grad[num_prev_tasks].dependencies.push_back(tasks_backward[num_prev_tasks].mymutex);
      ++num_prev_tasks;
      --bridge_id_no_model_p;
      --bridge_id_with_model_p;
    }
    
    assert(bridge_id_no_model_p == -1);
    assert(bridge_id_with_model_p == -1);
    
    // SHADJIS TODO: Make a final one which prints out time etc. and also fills msg --> I put in the UDF for now
    //  - In CC this was previously its own task
    
    
    // =============================================================================================
    // DONE TASKS
    // =============================================================================================

    size_t num_model_bridge_tasks = tasks_forward.size();
    assert(tasks_get_model.size()   == num_model_bridge_tasks);
    assert(tasks_forward.size()     == num_model_bridge_tasks);
    assert(tasks_backward.size()    == num_model_bridge_tasks);
    assert(tasks_update_grad.size() == num_model_bridge_tasks);
    
    // Create Queues
    TaskQueue queue;
    queue.tasks.push_back(task_get_fw_data);
    for(size_t i=0;i<tasks_get_model.size();i++){
      queue.tasks.push_back(tasks_get_model[i]);
    }
    for(size_t i=0;i<tasks_forward.size();i++){
      queue.tasks.push_back(tasks_forward[i]);
    }
    // queue.tasks.push_back(task_update_acc);
    for(size_t i=0;i<tasks_backward.size();i++){
      queue.tasks.push_back(tasks_backward[i]);
    }
    for(size_t i=0;i<tasks_update_grad.size();i++){
      queue.tasks.push_back(tasks_update_grad[i]);
    }
    // SHADJIS TODO: Final task to print loss etc (Inside UDF now)

    
    // Make a UDF which defines a task and runs it
    // This UDF will be called whenever the broker has new data
    auto UDF = [&](OmvMessage ** msgs, int nmsg, OmvMessage * & msg){
        for (int i=0; i< group_size; ++i) {
            tmp_msgs[i] = msgs[i];
        }
        tmp_nmsg = nmsg;
        assert(nmsg == group_size);
        queue.prepare();
        queue.run();
        
        // SHADJIS TODO: Can put rest of this into a task
        
        VLOG(2) << "~~~~ ENTER STATE ACC" << std::endl;
        loss += (softmax->get_loss() / float(corpus->mini_batch_size));
        accuracy += float(DeepNet::find_accuracy(labels, (*--bridges.end())->p_output_layer->p_data_cube)) / float(corpus->mini_batch_size);
        
        // Now need to fill msg
        
        // Check if we should print batch status
        if ( (batch+1) % display_iter == 0 ) {
          float learning_rate = Util::get_learning_rate(solver_param.lr_policy(), solver_param.base_lr(), solver_param.gamma(),
            batch+1, solver_param.stepsize(), solver_param.power(), solver_param.max_iter());
          
          std::cout << "Training Status Report (Mini-batch iter " << batch << "), LR = " << learning_rate << std::endl;
          std::cout << "  \033[1;32m";
          //std::cout << "Loss & Accuracy [Average of Past " << display_iter << " Iterations]\t" << loss/float(display_iter) << "\t" << float(accuracy)/(float(display_iter));
          std::cout << "Loss & Accuracy [Average of Past " << display_iter << " Iterations]\t" << timer.elapsed() << "\t" << loss/float(display_iter) << "\t" << float(accuracy)/(float(display_iter));
          std::cout << "\033[0m" << std::endl;
          loss = 0.;
          accuracy = 0.;
        }
        VLOG(2) << "~~~~ EXIT STATE ACC" << std::endl;
       
        // -----------------------------------------------------------------------
        // Fill outgoing_msg_data_grad->content with the data gradients
        // -----------------------------------------------------------------------
        // SHADJIS TODO: See comment above, this code needs to change if doing a
        // direct copy across servers to GPU memory to use device memory pointers.
        //assert(!bridges[0]->get_share_pointer_with_prev_bridge());
        VLOG(2) << "~~~~ ENTER STATE Send Back Data Grad" << std::endl;
        
        // SHADJIS TODO: Here I use a memcpy instead of just assigning the pointer?
        memcpy(outgoing_msg_data_grad->content, 
          bridges[0]->p_input_layer->p_gradient_cube->get_p_data(),
          sizeof(float)*nfloats_data);
        VLOG(2) << "~~~~ EXIT STATE Send Back Data Grad" << std::endl;
        
        msg = outgoing_msg_data_grad;
        
        // Check if we should write a snapshot
        if (snapshot > 0 && (batch+1) % snapshot == 0) {
          time_t rawtime;
          struct tm * timeinfo;
          char buffer[80];
          time (&rawtime);
          timeinfo = localtime(&rawtime);
          strftime(buffer,80,"%d-%m-%Y-%I-%M-%S",timeinfo);
          std::string str(buffer);
          std::string snapshot_name;
          snapshot_name = solver_file + "_MODEL." + str;
          DeepNet::write_model_to_file(bridges, snapshot_name);
          std::cout << "======= Writing snapshot " << snapshot_name << " =======" << std::endl;
        }
        
        ++ batch;
        
    };  // END UDF

    // Start this broker with the input ports
    Broker_N_1<decltype(UDF)> broker(conv_listen_bind, conv_send_bind, incoming_data_buf_size, outgoing_data_grad_buf_size, group_size);
    broker.start(UDF);

    // -------------------------------------------------------------------------
    // Destroy network
    // -------------------------------------------------------------------------
    DeepNet::clean_up(bridges, corpus);

  }
};

#endif

