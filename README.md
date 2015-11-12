# Distributed CcT

### Compilation

Download the repository as well as the CaffeConTroll submodule

  - `git clone https://github.com/HazyResearch/dcct.git`
  - `cd dcct`
  - `git submodule update --init`

Install the following dependencies

  - zeromq
     - The zeromq website seems to be out of date. If so, you can try [this helpful blog post](https://tuananh.org/2015/06/16/how-to-install-zeromq-on-ubuntu/)
  - Protobuf 3.0.0, e.g. using the command `pip install protobuf==3.0.0a3`
  - LMDB, e.g. using the command `pip install lmdb`
  - glog (this dependency may be removed soon)
  - [libconfig++](http://www.hyperrealm.com/libconfig/) (this dependency may be removed soon)
  
Compile dcct

    `make -j`
    `cd tools/size_util/; make -j; cd -`
    
This will produce a binary `dcct`.

### Usage

Create a file containing each machine in your cluster, one per line
  - See an example [here](example/machine_list.txt)

Create a solver and train prototxt file (same input as Caffe/CaffeConTroll) and run:

  - `python run.py  path/to/solver.prototxt  path/to/machine_list.txt`

Observing the output:

  - A number of configuration files will be created in a new directory called something like `server_input_files-2015-11-12-05-25-26`
  - The output of the network that is training will be written to the file `fc_server.cfg.out` inside this directory
