# Distributed CcT

### Compilation

Download the repository as well as the CaffeConTroll submodule

  - `git clone https://github.com/HazyResearch/dcct.git`
  - `cd dcct`
  - `git submodule update --init`

Install the following dependencies

  - zeromq
     - The zeromq website seems to be out of date. If so, you can try [this helpful blog post](https://tuananh.org/2015/06/16/how-to-install-zeromq-on-ubuntu/)
  - glog (this dependency may be removed soon, but currently it is a dependency for both CcT and dcct)
  - [libconfig++](http://www.hyperrealm.com/libconfig/) (this dependency may be removed soon)
  - Python library: Protobuf 3.0.0, e.g. using the command `pip install protobuf==3.0.0a3` (the C++ library still needs to be installed below)
  - Python library: LMDB, e.g. using the command `pip install lmdb` (the C++ library still needs to be installed below)

Also install the dependencies for the CaffeConTroll submodule (follow steps 1 and 3 [here](https://github.com/HazyResearch/CaffeConTroll/tree/experiments#installation-from-source))

Compile dcct

    make -j
    cd tools/size_util/; make -j; cd -
    
This will produce a binary `dcct`.

### Usage

Create a file containing each machine in your cluster, one per line

  - The machine name should be the same as the machine you would `ssh` into
  - For example if you would normally do: `ssh root@node012`, then your machine list would contain `node012`
  - See an example [here](example/machine_list.txt)

Create a solver and train prototxt file (same input as Caffe/CaffeConTroll)

Edit the top of the file [run.py](run.py), located in the root directory, to update the script parameters:

  - Edit the line `user = 'root'` in the `SSH parameters` section to add the right username for ssh
     - For the example above (`ssh root@node012`), the username put on that line would be `root`
  - Edit the line `extra_cmd = ...` in the `SSH parameters` section to add any lines needed between when you first `ssh` into your machine and when you are in the `dcct` directory about to run the `./dcct` executable (usually these are `cd` commands as well as loading libraries from your `~/.bashrc` file, which is not sourced when doing `ssh`)
  - (Optional) Edit the line `use_4_gpu` or `use_1_gpu` in the `Script parameters` section depending on how many GPUs your machines have
  - (Optional) To save time when running the script more than once, set the line `skip_lmdb_generation` to `True` after the first time the script runs. The LMDB needs to be partitioned for each machine, but this only needs to be done once. Note: if the number of machines changes, partitions need to be recreated so make sure to reset this to `False`

(in the future these will be inputs to the script `run.py`, e.g. through a config file.)
  
Run:

  - `python run.py  path/to/solver.prototxt  path/to/machine_list.txt`

Observing the output:

  - A number of configuration files will be created in a new directory called something like `server_input_files-2015-11-12-05-25-26`
  - The output of the network that is training will be written to the file `fc_server.cfg.out` inside this directory
