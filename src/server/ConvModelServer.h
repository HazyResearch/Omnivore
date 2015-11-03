
#ifndef _CONVMODELSERVER_H
#define _CONVMODELSERVER_H

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

class ConvModelServer : public Server{
public:

  string name;
  string bind;

  int nfloats;
  char * buf;

  using Server::input;
  using Server::output;
  using Server::models;

  ConvModelServer(string _name, string _bind, int _nfloats){
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
    LOG(INFO) << "Starting ConvModelServer[" << name << "]..." << std::endl;

    void *context = zmq_ctx_new ();
    void *responder = zmq_socket (context, ZMQ_REP);
    int rc = zmq_bind (responder, bind.c_str());
    assert (rc == 0);

    LOG(INFO) << "Binded to " << bind << std::endl;

    // for now, the buffer is 2x larger than the size of
    // the model. Don't see why this needs to be larger
    int bufsize = nfloats * sizeof(float) * 2;
    char * buffer = new char[nfloats * sizeof(float) * 2];

    Message * master_model_msg = reinterpret_cast<Message*>(buf);
    master_model_msg->msg_type = ANSWER_MODEL;
    master_model_msg->nelem = nfloats;

    Message reply_update_gradient;
    reply_update_gradient.msg_type = ANSWER_UPDATE_GRADIENT;
    reply_update_gradient.nelem = 0;

    while (1) {

      zmq_recv(responder, buffer, bufsize, 0);
      Message * msg = reinterpret_cast<Message*>(buffer);

      if(msg->msg_type == ASK_MODEL){
        // Reply Current Model
        LOG(INFO) << "Responding ASK_MODEL Request" << endl;
        zmq_send (responder, buf, master_model_msg->size(), 0);
      }else if(msg->msg_type == ASK_UPDATE_GRADIENT){   // TODO: NEED TO READ 
                                                        // PROTOCOLBUF AND DO RIGHT THING
        LOG(INFO) << "Updating Gradient" << endl;
        for(int i=0;i<msg->nelem;i++){                  // Simple Update of the Gradient
          master_model_msg->content[i] -= STEPSIZE * msg->content[i];
        }
        zmq_send (responder, &reply_update_gradient, reply_update_gradient.size(), 0);
      }else{
        LOG(WARNING) << "Ignore unsupported message type " << msg->msg_type << endl;
      }
    }

  }
};

#endif








