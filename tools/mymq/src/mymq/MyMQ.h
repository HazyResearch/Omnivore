#ifndef _MYMQ_H
#define _MYMQ_H

#include <string>
#include <zmq.h>
#include <assert.h>
#include <iostream>

#include <xmmintrin.h>
#include <ammintrin.h>

#include "mymq/OmvMessage.h"



void my_memcpy_32bitfloat_to_16bitfloat(void * _dst, void * _src, int nelem){

  uint16_t * dst = (uint16_t *) _dst;
  float * src = (float *) _src;

  int j=0;
  for(int i=0;i<nelem;i+=8){
    __m256 float_vector = _mm256_load_ps(&dst[i]);
    __m128i half_vector = _mm256_cvtps_ph(float_vector, 0);
    _mm_store_si128 ((__m128i*)halfs, dst[i/2]);
    j += 4;
  }

}


void my_memcpy_16bitfloat_to_32bitfloat(void * _dst, void * _src, int nelem){



}



enum CHANNELTYPE {
  CHANNEL_FULLPREC = 0,
  CHANNEL_16BITFLOAT = 1,
  CHANNEL_8BITFLOAT = 2
};

class MyMQ {
public:

  std::string bind;

  int socket_type;

  int max_buffer_in_byte;

  void *context;

  void *responder;

  bool isclient;

  char* buf;

  MyMQ(std::string _bind, int _socket_type, int _max_buffer_in_byte, bool _isclient) :
    bind(_bind), socket_type(_socket_type),
    max_buffer_in_byte(_max_buffer_in_byte),
    isclient(_isclient)
  {

    context = zmq_ctx_new ();
    responder = zmq_socket (context, socket_type);

    if(isclient){
      int rc = zmq_connect (responder, bind.c_str());
      assert (rc == 0);
    }else{
      int rc = zmq_bind (responder, bind.c_str());
      assert (rc == 0);
    }

    buf = new char[max_buffer_in_byte];
  }

  void recv(OmvMessage * msg, CHANNELTYPE channeltype){

    if(channeltype == CHANNEL_FULLPREC){
      zmq_recv(responder, (char*) msg, max_buffer_in_byte, 0);
    }else if(channeltype == CHANNEL_16BITFLOAT){
      zmq_recv(responder, buf, max_buffer_in_byte, 0);
      OmvMessage * recved_msg = (OmvMessage *) buf;
      memcpy(msg, recved_msg, sizeof(OmvMessage));
    }else if(channeltype == CHANNEL_8BITFLOAT){
      zmq_recv(responder, buf, max_buffer_in_byte, 0);
      OmvMessage * recved_msg = (OmvMessage *) buf;
      memcpy(msg, recved_msg, sizeof(OmvMessage));
    }else{
      assert(false);
    }

  }

  void send(OmvMessage * msg, CHANNELTYPE channeltype){

    if(channeltype == CHANNEL_FULLPREC){
      zmq_send(responder, (char*) msg, msg->size(), 0);
    }else if(channeltype == CHANNEL_16BITFLOAT){
      OmvMessage * sent_msg = (OmvMessage *) buf;
      memcpy(sent_msg, msg, sizeof(OmvMessage));
      my_memcpy_32bitfloat_to_16bitfloat(
        msg->content, sent_msg->content
        msg->nelem);
      zmq_send(responder, (char*) sent_msg, 
        sizeof(OmvMessage) + sizeof(float)/2*msg->nelem, 0);
    }else if(channeltype == CHANNEL_8BITFLOAT){
      OmvMessage * sent_msg = (OmvMessage *) buf;
      memcpy(sent_msg, msg, sizeof(OmvMessage));
      zmq_send(responder, (char*) sent_msg, 
        sizeof(OmvMessage) + sizeof(float)/4*msg->nelem, 0);
    }else{
      assert(false);
    }
  }

};


#endif


















