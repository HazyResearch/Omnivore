
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
#include "server/FCComputeServer.h"
#include "server/FCModelServer.h"

// SHADJIS TODO: We use libconfig++ to parse very simple files, 
// can just parse manually

using namespace std;
using namespace libconfig;


// Initialize conv model server from config file
Server * initConvModelServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing ConvModelServer from " << filename << endl;

  string NAME      = cfg.lookup("name");
  string SOLVER    = cfg.lookup("solver");
  string TRAIN_BIN = cfg.lookup("train_bin");
  int    GROUPSIZE = cfg.lookup("group_size");
  
  // Now parse each group of ports
  // Recall a conv model server has 2 ports per conv compute group
  // (1 port for broadcasting, 1 for listening)
  const Setting& root = cfg.getRoot();
  int num_groups = root["ports"].getLength();
  
  LOG(INFO) << "GROUPSIZE  = " << GROUPSIZE  << std::endl;
  LOG(INFO) << "NUM GROUPS = " << num_groups << std::endl;
  LOG(INFO) << "NAME       = " << NAME       << std::endl;
  LOG(INFO) << "SOLVER     = " << SOLVER     << std::endl;
  LOG(INFO) << "TRAIN_BIN  = " << TRAIN_BIN  << std::endl;

  // Read the ports
  std::vector <string> broadcast_ports;
  std::vector <string> listen_ports;
  for(int i=0;i<num_groups;i++){
    const Setting & port_pair = root["ports"][i];
    string broadcast;
    string listen;
    port_pair.lookupValue("broadcast", broadcast);
    port_pair.lookupValue("listen", listen);
    LOG(INFO) << "Group " << i << " listening on port " << listen << " and broadcasting on port " 
              << broadcast << std::endl;
    broadcast_ports.push_back(broadcast);
    listen_ports.push_back(listen);
  }

  Server * s = new ConvModelServer(NAME, SOLVER, TRAIN_BIN, GROUPSIZE,
    broadcast_ports, listen_ports);
  return s;
}


// Initialize fc model/compute server from config file
Server * initFCComputeModelServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing FCComputeModelServer from " << filename << endl;

  string NAME      = cfg.lookup("name");
  string SOLVER    = cfg.lookup("solver");
  string TRAIN_BIN = cfg.lookup("train_bin");
  int    GROUPSIZE = cfg.lookup("group_size");

  // Now parse each group of ports
  // Recall a conv model server has 2 ports per conv compute group
  // (1 port for broadcasting, 1 for listening)
  const Setting& root = cfg.getRoot();
  int num_groups = root["ports"].getLength();
  
  LOG(INFO) << "GROUPSIZE  = " << GROUPSIZE  << std::endl;
  LOG(INFO) << "NUM GROUPS = " << num_groups << std::endl;
  LOG(INFO) << "NAME       = " << NAME       << std::endl;
  LOG(INFO) << "SOLVER     = " << SOLVER     << std::endl;
  LOG(INFO) << "TRAIN_BIN  = " << TRAIN_BIN  << std::endl;

  // Read the ports
  std::vector <string> broadcast_ports;
  std::vector <string> listen_ports;
  for(int i=0;i<num_groups;i++){
    const Setting & port_pair = root["ports"][i];
    string broadcast;
    string listen;
    port_pair.lookupValue("broadcast", broadcast);
    port_pair.lookupValue("listen", listen);
    LOG(INFO) << "Group " << i << " listening on port " << listen << " and broadcasting on port " 
              << broadcast << std::endl;
    broadcast_ports.push_back(broadcast);
    listen_ports.push_back(listen);
  }

  Server * s = new FCComputeModelServer(NAME, SOLVER, TRAIN_BIN, GROUPSIZE,
    broadcast_ports, listen_ports);
  return s;
}


// Initialize fc compute server from config file
Server * initFCComputeServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing initFCComputeServer from " << filename << endl;

  string NAME             = cfg.lookup("name");
  string FC_LISTEN_BIND   = cfg.lookup("fc_listen_bind");
  string FC_SEND_BIND     = cfg.lookup("fc_send_bind");
  string CONV_LISTEN_BIND = cfg.lookup("conv_listen_bind");
  string CONV_SEND_BIND   = cfg.lookup("conv_send_bind");
  string SOLVER           = cfg.lookup("solver");
  string TRAIN_BIN        = cfg.lookup("train_bin");
  int    GROUPSIZE = cfg.lookup("group_size");
  int RANK_IN_GROUP= cfg.lookup("rank_in_group");

  LOG(INFO) << "NAME             = " << NAME             << std::endl;
  LOG(INFO) << "FC_LISTEN_BIND   = " << FC_LISTEN_BIND   << std::endl;
  LOG(INFO) << "FC_SEND_BIND     = " << FC_SEND_BIND     << std::endl;
  LOG(INFO) << "CONV_LISTEN_BIND = " << CONV_LISTEN_BIND << std::endl;
  LOG(INFO) << "CONV_SEND_BIND   = " << CONV_SEND_BIND   << std::endl;
  LOG(INFO) << "SOLVER           = " << SOLVER           << std::endl;
  LOG(INFO) << "TRAIN_BIN        = " << TRAIN_BIN        << std::endl;
  LOG(INFO) << "GROUPSIZE        = " << GROUPSIZE << std::endl;
  LOG(INFO) << "RANK             = " << RANK_IN_GROUP << std::endl;

  Server * s = new FCComputeServer(NAME, FC_LISTEN_BIND, FC_SEND_BIND,
    CONV_LISTEN_BIND, CONV_SEND_BIND, SOLVER, TRAIN_BIN, GROUPSIZE, RANK_IN_GROUP);
  return s;
}


// Initialize fc model server from config file
Server * initFCModelServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing initFCModelServer from " << filename << endl;

  string NAME      = cfg.lookup("name");
  string SOLVER    = cfg.lookup("solver");
  string TRAIN_BIN = cfg.lookup("train_bin");
  int    GROUPSIZE = cfg.lookup("group_size");

  // Now parse each group of ports
  // Recall a conv model server has 2 ports per conv compute group
  // (1 port for broadcasting, 1 for listening)
  const Setting& root = cfg.getRoot();
  int num_groups = root["ports"].getLength();
  
  LOG(INFO) << "GROUPSIZE  = " << GROUPSIZE  << std::endl;
  LOG(INFO) << "NUM GROUPS = " << num_groups << std::endl;
  LOG(INFO) << "NAME       = " << NAME       << std::endl;
  LOG(INFO) << "SOLVER     = " << SOLVER     << std::endl;
  LOG(INFO) << "TRAIN_BIN  = " << TRAIN_BIN  << std::endl;

  // Read the ports
  std::vector <string> broadcast_ports;
  std::vector <string> listen_ports;
  for(int i=0;i<num_groups;i++){
    const Setting & port_pair = root["ports"][i];
    string broadcast;
    string listen;
    port_pair.lookupValue("broadcast", broadcast);
    port_pair.lookupValue("listen", listen);
    LOG(INFO) << "Group " << i << " listening on port " << listen << " and broadcasting on port " 
              << broadcast << std::endl;
    broadcast_ports.push_back(broadcast);
    listen_ports.push_back(listen);
  }

  Server * s = new FCModelServer(NAME, SOLVER, TRAIN_BIN, GROUPSIZE,
    broadcast_ports, listen_ports);
  return s;
}


// Initialize conv compute server from config file
Server * initConvComputeServer(Config & cfg, char * filename){
  LOG(INFO) << "Initializing ConvComputeServer from " << filename << endl;

  string NAME             = cfg.lookup("name");
  string CONV_LISTEN_BIND = cfg.lookup("conv_listen_bind");
  string CONV_SEND_BIND   = cfg.lookup("conv_send_bind");
  string FC_LISTEN_BIND   = cfg.lookup("fc_listen_bind");
  string FC_SEND_BIND     = cfg.lookup("fc_send_bind");
  string SOLVER           = cfg.lookup("solver");
  string TRAIN_BIN        = cfg.lookup("train_bin");

  int    GROUPSIZE = cfg.lookup("group_size");
  int RANK_IN_GROUP= cfg.lookup("rank_in_group");

  LOG(INFO) << "NAME             = " << NAME             << std::endl;
  LOG(INFO) << "CONV_LISTEN_BIND = " << CONV_LISTEN_BIND << std::endl;
  LOG(INFO) << "CONV_SEND_BIND   = " << CONV_SEND_BIND   << std::endl;
  LOG(INFO) << "FC_LISTEN_BIND   = " << FC_LISTEN_BIND   << std::endl;
  LOG(INFO) << "FC_SEND_BIND     = " << FC_SEND_BIND     << std::endl;
  LOG(INFO) << "SOLVER           = " << SOLVER           << std::endl;
  LOG(INFO) << "TRAIN_BIN        = " << TRAIN_BIN        << std::endl;

  LOG(INFO) << "GROUPSIZE = " << GROUPSIZE << std::endl;
  LOG(INFO) << "RANK      = " << RANK_IN_GROUP << std::endl;

  Server * s = new ConvComputeServer(NAME, CONV_LISTEN_BIND, CONV_SEND_BIND, FC_LISTEN_BIND,
    FC_SEND_BIND, SOLVER, TRAIN_BIN, GROUPSIZE, RANK_IN_GROUP);
  return s;
}


// Read in path to config file and start server
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  Config cfg;
  
  // Expect config file as first and only argument
  if(argc < 2){
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
  }else if(type == "FCComputeServer"){
    Server * s = initFCComputeServer(cfg, argv[1]);
    s->start();
  }else if(type == "FCModelServer"){
    Server * s = initFCModelServer(cfg, argv[1]);
    s->start();
  }else{
    LOG(FATAL) << "Unsupported Server Type." << endl;
  }

  return(EXIT_SUCCESS);
}
