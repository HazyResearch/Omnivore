#include "mymq/MyMQ.h"
#include <iostream>
#include <unistd.h>

#include "timer.h"

int main(){
  std::cout << "CLIENT..." << std::endl;

  OmvMessage * msg = new OmvMessage;
  msg->nelem = 1000000;
  char * buf = new char[msg->size()];
  msg = (OmvMessage *) buf;
  msg->nelem = 1000000;

  MyMQ mymq("tcp://localhost:5555", ZMQ_REQ, 1000, true);
  float i = 1;
  while(1){
    msg->content[0] = i * 3.14;
    std::cout << msg->content[0] << std::endl;
    
    Timer t;
    mymq.send(msg, CHANNEL_16BITFLOAT);
    std::cout << "TIME: " << t.elapsed() << std::endl;
    std::cout << "THROUGHPUT: " << 
      (1.0*msg->size()/1024/1024/t.elapsed())
      << " MB/s" << std::endl;

    mymq.recv(msg, CHANNEL_16BITFLOAT);
    sleep(2);
    i++;
  }

  return 0;
}