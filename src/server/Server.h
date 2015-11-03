#ifndef _SERVER_H
#define _SERVER_H

#include <vector>
#include "dstruct/Cube.h"

using namespace std;

class Server {
public:

  virtual void start() = 0;

  Cube input;

  Cube output;
  
  vector<Cube> models;

};










#endif