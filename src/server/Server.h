#ifndef _SERVER_H
#define _SERVER_H

class Server {
public:
  virtual void start() = 0;
  std::string name;
  std::string solver_file;
  std::string data_binary;
};

#endif
