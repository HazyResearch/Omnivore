
#ifndef _FCCOMPUTEMODELSERVER_H
#define _FCCOMPUTEMODELSERVER_H

#include <iostream>
#include <string>
#include <cstring>
#include <glog/logging.h>

#include "server/Server.h"
#include "message/Message.h"

#include <zmq.h>

using namespace std;
using namespace libconfig;

#define STEPSIZE 0.01

class FCComputeModelServer : public Server{
public:

  string name;
  string bind;

  int nfloats;
  char * buf;

  int nfloats_input_data;
  int nfloats_input_grad;

  using Server::input;
  using Server::output;
  using Server::models;

  FCComputeModelServer(string _name, string _bind, int _nfloats){
    name = _name;
    bind = _bind;
    nfloats = _nfloats;
    LOG(INFO) << "Allocating " << (1.0*nfloats*sizeof(float)/1024/1024) << " MB" << std::endl;
    buf = new char[sizeof(Message) + sizeof(float) * nfloats];
  }

  /**
   * A ConvModelServer does two things
   *   - listen to request that asks model.
   *   - listen to response that returns the gradient.
   **/
  void start(){
    LOG(INFO) << "Starting FCComputeModelServer[" << name << "]..." << std::endl;

    nfloats_input_data = input.N*input.N*input.I*input.B;
    nfloats_input_grad = nfloats_input_data;

    void *context = zmq_ctx_new ();
    void *responder = zmq_socket (context, ZMQ_REP);
    int rc = zmq_bind (responder, bind.c_str());
    assert (rc == 0);

    LOG(INFO) << "Binded to " << bind << std::endl;

    // for now, the buffer is 2x larger than the size of
    // the model. Don't see why this needs to be larger
    int bufsize = nfloats * sizeof(float) * 2;
    char * buffer = new char[nfloats * sizeof(float) * 2];

    Message * gradient_msg = reinterpret_cast<Message*>(buf);
    gradient_msg->msg_type = ANSWER_GRADIENT;
    gradient_msg->nelem = nfloats_input_data;

    while (1) {

      zmq_recv(responder, buffer, bufsize, 0);
      Message * msg = reinterpret_cast<Message*>(buffer);

      if(msg->msg_type == ASK_GRADIENT){
        LOG(INFO) << "Responding ASK_GRADIENT Request" << endl;
        
        cout << msg->content[0] << endl;    // Print the first element of the data

        sleep(1);                           // TODO: Do something and fill in gradient with -2
        for(int i=0;i<gradient_msg->nelem;i++){
          gradient_msg->content[i] = -2;
        }

        zmq_send (responder, gradient_msg, gradient_msg->size(), 0);
      }else{
        LOG(WARNING) << "Ignore unsupported message type " << msg->msg_type << endl;
      }
    }

  }
};

#endif








