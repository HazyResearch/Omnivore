#ifndef _MESSAGE_H
#define _MESSAGE_H

enum MessageType {
  
  ASK_MODEL = 0,
  ASK_UPDATE_GRADIENT = 1,
  ASK_GRADIENT = 2,

  ANSWER_MODEL = 10,
  ANSWER_UPDATE_GRADIENT = 11,
  ANSWER_GRADIENT = 12,
};


struct Message {
  
  MessageType msg_type;

  int nelem;

  float content[0];

  int size(){
    return sizeof(Message) + nelem*sizeof(float);
  }

};



#endif