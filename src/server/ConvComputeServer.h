#ifndef _CONVCOMPUTESERVER_H
#define _CONVCOMPUTESERVER_H

#include <iostream>
#include <string>
#include <cstring>
#include <glog/logging.h>

#include "server/Server.h"
#include "message/Message.h"

#include <zmq.h>

using namespace std;
using namespace libconfig;

class ConvComputeServer : public Server{
public:

  string name;
  string conv_bind;
  string fc_bind;
  int nfloats;

  int nfloats_output_data;

  int nfloats_output_grad;

  using Server::input;
  using Server::output;
  using Server::models;

  ConvComputeServer(string _name, string _conv_bind, string _fc_bind, int _nfloats){
    name = _name;
    conv_bind = _conv_bind;
    fc_bind = _fc_bind;
    nfloats = _nfloats;
  }

  void start(){
    LOG(INFO) << "Starting ConvComputeServer[" << name << "]..." << std::endl;

    nfloats_output_data = output.N*output.N*output.I*output.B;
    nfloats_output_grad = nfloats_output_data;

    void *context = zmq_ctx_new ();
    void *requester = zmq_socket (context, ZMQ_REQ);
    zmq_connect (requester, conv_bind.c_str());

    void *context_fc = zmq_ctx_new ();
    void *requester_fc = zmq_socket (context_fc, ZMQ_REQ);
    zmq_connect (requester_fc, fc_bind.c_str());

    // For input models

    LOG(INFO) << "Allocating " << (1.0*nfloats*sizeof(float)/1024/1024) << " MB for model" << std::endl;
    int bufsize = nfloats * sizeof(float) * 2;
    char * model_buffer = new char[nfloats * sizeof(float) * 2];
    char * grad_buffer = new char[nfloats * sizeof(float) * 2];

    Message * grad_msg = reinterpret_cast<Message*>(grad_buffer);
    grad_msg->msg_type = ASK_UPDATE_GRADIENT;
    grad_msg->nelem = nfloats;

    Message ask_model_msg;
    ask_model_msg.msg_type = ASK_MODEL;
    ask_model_msg.nelem = 0;

    Message answer_update_grad;
    ask_model_msg.msg_type = ASK_MODEL;
    ask_model_msg.nelem = 0;

    // For output data

    LOG(INFO) << "Allocating " << (1.0*nfloats_output_data*sizeof(float)/1024/1024) << " MB for output data" << std::endl;
    int out_bufsize = nfloats_output_data * sizeof(float) * 2;
    char * out_data_buffer = new char[out_bufsize];
    char * out_grad_buffer = new char[out_bufsize];

    Message * output_ask_grad_msg = reinterpret_cast<Message*>(out_data_buffer);
    output_ask_grad_msg->msg_type = ASK_GRADIENT;
    output_ask_grad_msg->nelem = nfloats_output_data;
    
    while(1){
      // first send request asking for model
      LOG(INFO) << "Ask Current Model..." << std::endl;
      zmq_send (requester, &ask_model_msg, ask_model_msg.size(), 0);

      // second, receive model
      zmq_recv (requester, model_buffer, bufsize, 0);
      Message * modelmsg = reinterpret_cast<Message*>(model_buffer);
      assert(modelmsg->msg_type == ANSWER_MODEL);

      // third, do some work
      LOG(INFO) << "Do Real Work..." << std::endl;  // TODO: BRIDGE WITH CCT OR CAFFE
      
      cout << modelmsg->content[0] << endl;// Print the first element of the model

      sleep(1);                            // Sleep 1 second (like the forward loop)
      for(int i=0;i<output_ask_grad_msg->nelem;i++){
        output_ask_grad_msg->content[i] = 100;
      }
      cout << "~" << output_ask_grad_msg->content[0] << std::endl;

                                           // send data to FC server
      LOG(INFO) << "   Ask grad of output by sending data" << endl;
      zmq_send (requester_fc, output_ask_grad_msg, output_ask_grad_msg->size(), 0);

                                           // receive grad from FC server
      zmq_recv (requester_fc, out_grad_buffer, out_bufsize, 0);
      LOG(INFO) << "   Rcv'ed grad of output" << endl;
      Message * output_rcv_grad_msg = reinterpret_cast<Message*>(out_grad_buffer);

      std::cout << output_rcv_grad_msg->content[0] << std::endl;

      sleep(1);                            // run backward prop
      for(int i=0;i<grad_msg->nelem;i++){  // Fill gradient with all -1
        grad_msg->content[i] = -1;
      }

      // forth, returns the gradient
      LOG(INFO) << "Update Gradient..." << std::endl;
      zmq_send (requester, grad_msg, grad_msg->size(), 0);

      // fifth, wait update done.
      zmq_recv (requester, &answer_update_grad, answer_update_grad.size(), 0);
      assert(answer_update_grad.msg_type == ANSWER_UPDATE_GRADIENT);
    }

  }
};

#endif










