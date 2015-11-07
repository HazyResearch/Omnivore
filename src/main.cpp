
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <assert.h>
#include <string>
#include <cstring>
#include <libconfig.h++>
#include <glog/logging.h>
#include <vector>

#include "server/Server.h"
#include "server/ConvModelServer.h"
#include "server/ConvComputeServer.h"
#include "server/FCComputeModelServer.h"

// SHADJIS TODO: This function used to read in .cfg files and create
// the server objects. Now, they read in their own proto files, so
// all this needs to do is pass the name of the proto.
// This will eventually be replaced with the scheduler: it takes in
// a solver/train proto like caffe, and splits it into partial protos
// based on sizes/etc.

using namespace std;
using namespace libconfig;

Server * initConvModelServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing ConvModelServer from " << filename << endl;
  const Setting& root = cfg.getRoot();

  string NAME = cfg.lookup("name");
  string BIND = cfg.lookup("bind");
  int NMODEL = root["models"].getLength();
  LOG(INFO) << "NAME   = " << NAME   << std::endl;
  LOG(INFO) << "BIND   = " << BIND   << std::endl;
  LOG(INFO) << "NMODEL = " << NMODEL << std::endl;

  int NNUMBERS = 0;

  for(int i=0;i<NMODEL;i++){
    const Setting & model = root["models"][i];
    string name;
    int N, I, B;
    model.lookupValue("name", name);
    model.lookupValue("N", N);
    model.lookupValue("I", I);
    model.lookupValue("B", B);
    LOG(INFO) << name << " " << N << " " << I << " " << B << endl;
    NNUMBERS += N * N * I * B;
  }
  LOG(INFO) << "NELEMS = " << NNUMBERS << " FLOATS" << std::endl;

  Server * s = new ConvModelServer(NAME, BIND); // SHADJIS TODO: Pass solver proto here

  return s;
}

Server * initConvComputeServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing ConvComputeServer from " << filename << endl;
  const Setting& root = cfg.getRoot();

  string NAME = cfg.lookup("name");
  string CONVBIND = cfg.lookup("conv_bind");
  string FCBIND = cfg.lookup("fc_bind");
  int NMODEL = root["models"].getLength();
  LOG(INFO) << "NAME    = " << NAME   << std::endl;
  LOG(INFO) << "CONVBIND= " << CONVBIND << std::endl;
  LOG(INFO) << "FCBIND  = " << FCBIND   << std::endl;
  LOG(INFO) << "NMODEL  = " << NMODEL << std::endl;

  int NNUMBERS = 0;

  vector<Cube> models;

  for(int i=0;i<NMODEL;i++){
    const Setting & model = root["models"][i];
    string name;
    int N, I, B;
    model.lookupValue("name", name);
    model.lookupValue("N", N);
    model.lookupValue("I", I);
    model.lookupValue("B", B);
    LOG(INFO) << name << " " << N << " " << I << " " << B << endl;
    NNUMBERS += N * N * I * B;
    models.push_back(Cube(name, N, I, B));
  }
  LOG(INFO) << "NELEMS = " << NNUMBERS << " FLOATS" << std::endl;

  Server * s = new ConvComputeServer(NAME, CONVBIND, FCBIND);   // SHADJIS TODO: Pass solver proto here
  
  string name;
  int N, I, B;
  root["input"].lookupValue("name", name);
  root["input"].lookupValue("N", N);
  root["input"].lookupValue("I", I);
  root["input"].lookupValue("B", B);
  s->input = Cube(name, N, I, B);

  root["output"].lookupValue("name", name);
  root["output"].lookupValue("N", N);
  root["output"].lookupValue("I", I);
  root["output"].lookupValue("B", B);
  s->output = Cube(name, N, I, B);

  s->models = models;
  return s;
}


Server * initFCComputeModelServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing initFCComputeModelServer from " << filename << endl;
  const Setting& root = cfg.getRoot();

  string NAME = cfg.lookup("name");
  string BIND = cfg.lookup("bind");
  int NMODEL = root["models"].getLength();
  LOG(INFO) << "NAME   = " << NAME   << std::endl;
  LOG(INFO) << "BIND   = " << BIND   << std::endl;
  LOG(INFO) << "NMODEL = " << NMODEL << std::endl;

  int NNUMBERS = 0;

  vector<Cube> models;

  for(int i=0;i<NMODEL;i++){
    const Setting & model = root["models"][i];
    string name;
    int N, I, B;
    model.lookupValue("name", name);
    model.lookupValue("N", N);
    model.lookupValue("I", I);
    model.lookupValue("B", B);
    LOG(INFO) << name << " " << N << " " << I << " " << B << endl;
    NNUMBERS += N * N * I * B;
    models.push_back(Cube(name, N, I, B));
  }
  LOG(INFO) << "NELEMS = " << NNUMBERS << " FLOATS" << std::endl;

  Server * s = new FCComputeModelServer(NAME, BIND);    // SHADJIS TODO: Pass solver proto here

  string name;
  int N, I, B;
  root["input"].lookupValue("name", name);
  root["input"].lookupValue("N", N);
  root["input"].lookupValue("I", I);
  root["input"].lookupValue("B", B);
  s->input = Cube(name, N, I, B);

  root["output"].lookupValue("name", name);
  root["output"].lookupValue("N", N);
  root["output"].lookupValue("I", I);
  root["output"].lookupValue("B", B);
  s->output = Cube(name, N, I, B);

  s->models = models;

  return s;
}


int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  Config cfg;

  if(argc < 2){
    LOG(FATAL) << "I need to know where to find config file as the first argument."
               << endl;
  }

  try{
    cfg.readFile(argv[1]);
  }catch(const FileIOException &fioex){
    LOG(FATAL) << "I/O error while reading file." << endl;
  }catch(const ParseException &pex){
    LOG(FATAL) << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << endl;
    assert(false);
  }

  // Get the Type
  string type = cfg.lookup("type");
  
  if(type == "ConvModelServer"){
    Server * s = initConvModelServer(cfg, argv[1]);
    s->start();
  }else if(type == "ConvComputeServer"){
    Server * s = initConvComputeServer(cfg, argv[1]);
    s->start();
  }else if(type == "FCComputeModelServer"){
    Server * s = initFCComputeModelServer(cfg, argv[1]);
    s->start();
  }else{
    LOG(FATAL) << "Unsupported Server Type." << endl;
  }

  return(EXIT_SUCCESS);
}
