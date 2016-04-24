#ifndef _MESSAGE_H
#define _MESSAGE_H

enum OmvMessageType {
  
  // Messages that ask for something
  // (if received, need to send something. If sent, wait to receive something)
  ASK_MODEL = -1,                        // This sends nothing, waits for a model
  ASK_UPDATE_GRADIENT = 1,              // This sends model gradients, waits for acknowledgement
  ASK_GRADIENT_OF_SENT_DATA = 2,        // This sends data, waits for data gradients
  
  // Messages that respond to ASK_ messages
  // (if sent, you are responding to an ASK. If received, you get the response of your ASK)
  ANSWER_MODEL = 10,                    // Respond to ASK_MODEL by sending model
  ANSWER_UPDATE_GRADIENT = 11,          // Respond to ASK_UPDATE_GRADIENT by sending ack
  ANSWER_GRADIENT_OF_SENT_DATA = 12,    // Respond to ASK_GRADIENT_AND_SEND_DATA by sending the data gradients

};


struct OmvMessage {
  
  OmvMessageType msg_type;

  int nelem;

  int bridgeid;

  int group_size;

  int rank_in_group;

  float content[0];

  int size(){
    return sizeof(OmvMessage) + nelem*sizeof(float);
  }

};



#endif