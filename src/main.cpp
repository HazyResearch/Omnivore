
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

// SHADJIS TODO: We use libconfig++ to parse very simple files, 
// can just parse manually

using namespace std;
using namespace libconfig;


// Initialize conv model server from config file
Server * initConvModelServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing ConvModelServer from " << filename << endl;

  string NAME      = cfg.lookup("name");
  string BIND      = cfg.lookup("bind");
  string SOLVER    = cfg.lookup("solver");
  string TRAIN_BIN = cfg.lookup("train_bin");

  LOG(INFO) << "NAME      = " << NAME      << std::endl;
  LOG(INFO) << "BIND      = " << BIND      << std::endl;
  LOG(INFO) << "SOLVER    = " << SOLVER    << std::endl;
  LOG(INFO) << "TRAIN_BIN = " << TRAIN_BIN << std::endl;

  Server * s = new ConvModelServer(NAME, BIND, SOLVER, TRAIN_BIN);
  return s;
}


// Initialize fc model/compute server from config file
Server * initFCComputeModelServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing initFCComputeModelServer from " << filename << endl;

  string NAME      = cfg.lookup("name");
  string BIND      = cfg.lookup("bind");
  string SOLVER    = cfg.lookup("solver");
  string TRAIN_BIN = cfg.lookup("train_bin");

  LOG(INFO) << "NAME      = " << NAME      << std::endl;
  LOG(INFO) << "BIND      = " << BIND      << std::endl;
  LOG(INFO) << "SOLVER    = " << SOLVER    << std::endl;
  LOG(INFO) << "TRAIN_BIN = " << TRAIN_BIN << std::endl;

  Server * s = new FCComputeModelServer(NAME, BIND, SOLVER, TRAIN_BIN);
  return s;
}


// Initialize conv compute server from config file
Server * initConvComputeServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing ConvComputeServer from " << filename << endl;

  string NAME      = cfg.lookup("name");
  string CONV_BIND = cfg.lookup("conv_bind");
  string FC_BIND   = cfg.lookup("fc_bind");
  string SOLVER    = cfg.lookup("solver");
  string TRAIN_BIN = cfg.lookup("train_bin");

  int    GROUPSIZE = cfg.lookup("group_size");
  int RANK_IN_GROUP= cfg.lookup("rank_in_group");

  LOG(INFO) << "NAME      = " << NAME      << std::endl;
  LOG(INFO) << "CONV_BIND = " << CONV_BIND << std::endl;
  LOG(INFO) << "FC_BIND   = " << FC_BIND   << std::endl;
  LOG(INFO) << "SOLVER    = " << SOLVER    << std::endl;
  LOG(INFO) << "TRAIN_BIN = " << TRAIN_BIN << std::endl;

  LOG(INFO) << "GROUPSIZE = " << GROUPSIZE << std::endl;
  LOG(INFO) << "RANK      = " << RANK_IN_GROUP << std::endl;

  Server * s = new ConvComputeServer(NAME, CONV_BIND, FC_BIND, SOLVER, TRAIN_BIN, GROUPSIZE, RANK_IN_GROUP);
  return s;
}


// Read in path to config file and start server
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  Config cfg;
  
  // Expect config file as first and only argument
  if(argc != 2){
    LOG(FATAL) << "Error: Expecting path to config file as the first and only argument."
               << endl;
  }
  
  // Parse server type from config file
  try{
    cfg.readFile(argv[1]);
  }catch(const FileIOException &fioex){
    LOG(FATAL) << "I/O error while reading file." << endl;
  }catch(const ParseException &pex){
    LOG(FATAL) << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << endl;
    assert(false);
  }
  string type = cfg.lookup("type");
  
  // Start server of this type
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
