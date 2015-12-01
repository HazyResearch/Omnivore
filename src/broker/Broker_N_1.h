#ifndef _BROKER_N_1
#define _BROKER_N_1

#include <iostream>
#include <string>
#include <cstring>
#include <glog/logging.h>
#include "server/Server.h"
#include "message/OmvMessage.h"
#include <zmq.h>
#include <assert.h>


/**
 * A N-1 Broker sits on the server, which is connected by a 
 * set of clients. One round of communication acts as the follows.
 *
 * 1. Aggregation
 *  Each client sends message of TYPE1 to the server.
 *  Server stacks these messages in a buffer, return ACK to client.
 *  Client receive ACK, then busy wait.
 *
 * 2. Synchronization
 *  Server busy wait until the buffer receives all messages from the client.
 *
 * 3. Broadcasting
 *  Server broadcasts messages of TYPE2 to the client. 
 *  Server then goes to 1.
 *
 * Therefore a N-1 broker needs:
 *  - Size of message of TYPE1.
 *  - UDF to gets from the BUFFER a message of TYPE2
 *  - A port for the collect port.
 *  - A port for the broadcast port.
 *
 **/
template<typename FuncType>
class Broker_N_1{
public:

  std::string bind_aggregator;
  std::string bind_broadcastor;

  int size_message_type1;
  int size_message_type2;

  int n_message_type1;

  OmvMessage ** messages_type1;

  OmvMessage * message_type2;

  char * messagebuf_type2;

  char * messagebuf_stacked_type1;

  Broker_N_1(
    std::string _bind_aggregator,
    std::string _bind_broadcastor,
    int _size_message_type1,
    int _size_message_type2,
    int _n_message_type1): 
  bind_aggregator(_bind_aggregator),
  bind_broadcastor(_bind_broadcastor),
  size_message_type1(_size_message_type1),
  size_message_type2(_size_message_type2),
  n_message_type1(_n_message_type1){
    messagebuf_stacked_type1 = new char[n_message_type1*size_message_type1];
    messages_type1 = new OmvMessage*[n_message_type1];
    messagebuf_type2 = new char[size_message_type2];
    message_type2 = reinterpret_cast<OmvMessage*>(messagebuf_type2);
  }

  void start(FuncType UDF){
    // bind two ports, where 
    //  - AGGREGATOR: REQ-REP PORT
    //  - BROADCASTOR: BROADCASTOR
    void * zmq_aggregator = my_create_zmq(bind_aggregator, ZMQ_REP);
    void * zmq_broadcastor= my_create_zmq(bind_broadcastor, ZMQ_PUB);

    // std::cout << "START" << std::endl;

    while(1){
      // Receive n_message_type1 msgs from the aggregation port
      VLOG(2) << "~~~~~" << "START RECEIVE" << std::endl;
      for(int n_got_type1=0;n_got_type1<n_message_type1;n_got_type1++){
        VLOG(2) << "        " << "START RECEIVE " << n_got_type1 << std::endl;
        zmq_recv(zmq_aggregator, &messagebuf_stacked_type1[n_got_type1*size_message_type1], 
                  size_message_type1, 0);
        OmvMessage * msg = reinterpret_cast<OmvMessage*>(
          &messagebuf_stacked_type1[n_got_type1*size_message_type1]);
        messages_type1[msg->rank_in_group] = msg;
        VLOG(2) << "        " << "RECEIVED RANK " << msg->rank_in_group << std::endl;
        zmq_send(zmq_aggregator, messagebuf_type2, 1, 0); // send dummy reply back
        VLOG(2) << "        " << "ACK RANK " << msg->rank_in_group << std::endl;
      }
      VLOG(2) << "~~~~~" << "FINISH RECEIVE" << std::endl;

      // Call UDFs to generate msg type 2
      UDF(messages_type1, n_message_type1, message_type2);

      VLOG(2) << "~~~~~" << "START BROADCAST" << std::endl;
      // Broadcast msg type 2,
      zmq_send (zmq_broadcastor, message_type2, message_type2->size(), 0); 
      //VLOG(2) << "SKIP" << std::endl;

      VLOG(2) << "~~~~~" << "FINISH BROADCAST" << std::endl;
    }
    
  }

private:

  void* my_create_zmq(std::string bind, int socket_type){
    VLOG(2) << "Creating SOCKET TYPE " << socket_type << " ON " << bind << std::endl;
    void *context = zmq_ctx_new ();
    void *responder = zmq_socket (context, socket_type);
    int rc = zmq_bind (responder, bind.c_str());
    assert (rc == 0);
    return responder;
  }

};

#endif

