# Omnivore Optimizer and Distributed CcT Prototype System

### Compilation

Download the repository as well as the CaffeConTroll submodule

  - `git clone https://github.com/HazyResearch/Omnivore.git`
  - `cd Omnivore`
  - `git submodule update --init`

Install the following dependencies

  - zeromq
     - The zeromq website seems to be out of date. If so, you can try [this helpful blog post](https://tuananh.org/2015/06/16/how-to-install-zeromq-on-ubuntu/)
  - glog
  - [libconfig++](http://www.hyperrealm.com/libconfig/)
  - Python library: Protobuf 3.0.0, e.g. using the command `pip install protobuf==3.0.0a3` (the C++ library still needs to be installed below)
  - Python library: LMDB, e.g. using the command `pip install lmdb` (the C++ library still needs to be installed below)

Also install the dependencies for the CaffeConTroll submodule (follow steps 1 and 3 [here](https://github.com/HazyResearch/CaffeConTroll/tree/experiments#installation-from-source))

Compile dcct (the distributed CcT system)

    make -j
    cd tools/size_util/; make -j; cd -
    
This will produce a binary `dcct`.

### Setup

Create a file containing each machine in your cluster, one per line

  - The machine name should be the same as the machine you would `ssh` into
  - For example if you would normally do: `ssh root@node012`, then your machine list would contain `node012`
  - See an example [here](example/machine_list.txt)

Create a solver and train prototxt file (same input as Caffe/CaffeConTroll)

Edit the top of the file [run.py](run.py), located in the root directory, to update the script parameters:

  - Edit the line `user = 'root'` in the `SSH parameters` section to add the right username for ssh
     - For the example above (`ssh root@node012`), the username put on that line would be `root`
  - Edit the line `extra_cmd = ...` in the `SSH parameters` section to add any lines needed between when you first `ssh` into your machine and when you are in the `Omnivore` directory about to run `./omnivore.py` (usually these are `cd` commands)

Edit the top of the file [omnivore.py](omnivore.py), located in the root directory, to update the script parameters:

  - Edit the line `base_dir = ` to a directory name which will hold the output of your next optmizer run (e.g. `base_dir = /home/Omnivore/imgnet1k_resnet`)
  - Edit the line `hw_type = ` to the type of hardware on each node, either `CPU`, `GPU` or `4GPU`
  - The first time you run `omnivore.py` on a new dataset and cluster, edit the line `MAKE_LMDB_FIRST = False` to make it `True`. This will partition the LMDB before the first optimizer run. Once this happens the first time, set `MAKE_LMDB_FIRST = False` so that future runs of `omnivore.py` do not need to repeat this step. If the number of machines changes, partitions need to be recreated.

### Usage

Run:

  - `./omnivore.py  path/to/config.file`

Example

  - `./omnivore.py  example/imgnet_1000.config`

Observing the output:

  - Sub-directories will be made for each optimizer run: first the cold start run, `COLD`, followed by each optimizer run, `OPT1`, `OPT2`, etc. Each of these runs consists of exploratory phases followed by the decision (final choice) for that run. 
  - Within each sub-directory, a number of configuration files will be created in directories called something like `server_input_files-2015-11-12-05-25-26`
  - The output of the network that is training will be written to the file `fc_server.cfg.out` inside this directory

### Advanced

  - `omnivore.py` runs the full optimizer. Individual dcct runs can be run using `run.py`
