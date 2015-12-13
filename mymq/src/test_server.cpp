#include "mymq/MyMQ.h"
#include <iostream>
#include <unistd.h>

int main(){
  std::cout << "SERVER..." << std::endl;

  OmvMessage * msg = new OmvMessage;
  msg->nelem = 1000000;
  char * buf = new char[msg->size()];
  msg = (OmvMessage *) buf;

  MyMQ mymq("tcp://*:5555", ZMQ_REP, 1000000, false);
  while(1){
    mymq.recv(msg, CHANNEL_16BITFLOAT);
    std::cout << msg->content[0] << std::endl;
    mymq.send(msg, CHANNEL_16BITFLOAT);
  }

  return 0;
}