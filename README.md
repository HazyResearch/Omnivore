# Distributed CcT

### Compilation

Install
  - zeromq
  - glog
  - libconfig++

Then just
    `make`
    
This will produce a binary `dcct`.

### Usage

There are currently three types of servers
  - ConvModelServer: the master server that provides models for CONV layers.
  - ConvComputeServer: the slave server that gets model from `ConvModelServer` and returns the gradient.
  - FCComputeModelServer: the master+slave sver that gets data of CONV result from `ConvComputeServer` and returns the gradient.

Each server is specified with a configuration file. To start these servers
  - `./dcct configs/ConvModelServer.cfg`
  - `./dcct configs/ConvComputeServer.cfg`
  - `./dcct configs/FCComputeModelServer.cfg`

You can keep adding `ConvComputeServer.cfg` with the same command. 
