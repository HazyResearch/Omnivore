# ==============================================================================
# 
# run.py
# 
# usage:
# 
# $ python run.py  path/to/solver.prototxt  path/to/machine_list.txt  machines_per_batch  CPU|1GPU|4GPU  single_fc|many_fc    (map_fcc_to_cc=)1|0
# 
# ==============================================================================

import sys
import os
import re   # Regex
import subprocess
import datum_pb2
import lmdb
import datetime


# ==============================================================================
# Parameters
# ==============================================================================

# ------------------------------------------------------------------------------
# SSH parameters
# ------------------------------------------------------------------------------

# User name to log in to each machine with
user = 'root'

# Extra commands to run following ssh
# This should include:
#  - cd into correct directory
#  - path commands in .bashrc (ssh does not source .bashrc so its load libary 
#    commands may need to be included here, see the example below)
extra_cmd = 'cd ../home/software/dcct/; export PATH=$PATH:/usr/local/cuda-7.0/bin; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.0/lib64;'


# ------------------------------------------------------------------------------
# Script parameters
# ------------------------------------------------------------------------------

# Set to true after lmdb has been generated once (saves time)
# Warning: When changing the number of machines, need to reset this to True
skip_lmdb_generation = False

# Run with GPUs. If neither is True, uses CPU only.
# This is for both conv compute and fc compute/model servers
# For the fc compute server (case when single_FC_server = False), set num_gpu_per_fcc
# These will eventually not be bools but rather ints selected by the optimizer
# (for now, only support 0, 1, or 4 GPUs). Set both to false to use CPU only.
use_4_gpu = True   # Takes precedence if both this and use_1_gpu are True
use_1_gpu = False

# If using > 1 GPU, do model parallelism on FC
# This applies regardless of whether fc compute and model are one server or
# separate servers
multi_gpu_model_parallelism = True  # Will be default true later or selected by optimizer , e.g. turn off if fcc / fcm


# FC Compute / Model Server Parameters
single_FC_server = True

# The remaining parameters are only valid if single_FC_server is False
# For FC, the scheduler may choose to have a single fc compute + model server ("fccm",
# or the case when single_FC_server = True), or to have one or more fc compute ("fcc")
# servers as well as a separate fc model ("fcm") server.
#  - If there is a single server for both fc model and fc compute (single_FC_server = True),
#    then this server will use a number of gpus decided by use_4_gpu and use_1_gpu above 
#    (only 0, 1 or 4 GPUs per server supported). Then the parameters below are not used.
#  - If there are separate servers for fc compute and fc model (single_FC_server = False),
#    then for now fc model will be its own machine and fc computes will be spread across
#    the remaining machines. However, it is possible to have many fc compute (fcc) on 1 
#    machine, if and only if that machine has multiple GPUs. E.g. if we have 4 GPUs/machine,
#    there are 3 cases:
#       - 1 fc compute server on that machine, using either 1 GPU, 4 GPUs or CPU only
#       - 2 fc compute servers on that machine, using up to 2 GPUs each (or each can use 1 GPU)
#       - 4 fc compute servers on that machine, each using exactly 1 GPU (none will use CPU)

# Now, use_1_gpu and use_4_gpu will be IGNORED for FC, and applied only to conv. The params
# below take precedence for FC.
num_fcc_per_machine = 1
num_gpu_per_fcc_machine = 4


# ==============================================================================
# Description
# ==============================================================================
"""
--------------------------------------------------------------------------------
Input: 
--------------------------------------------------------------------------------
- List of machines


--------------------------------------------------------------------------------
Task:
--------------------------------------------------------------------------------
- Create input protobuf files (solver and train_val) for each server
    - Read in a machine list so we know how many conv compute servers to create
    - Parse the solver prototxt to get the name of the train prototxt file
    - Parse the train prototxt file for lmdb names
    - Parse the network train prototxt again, now to split into 2 networks
    - Create new solver and network prototxt files for each server
- Partition the lmdb for each server
- Create config files for each server
- Run ssh commands on each server


--------------------------------------------------------------------------------
Scheduler tradeoffs:
--------------------------------------------------------------------------------

- Machine types are conv model, conv compute, fc compute, fc model, and also
  combinations: conv compute/model (unimplemented), fc compute/model, 
  model server (unimplemented) and compute server (unimplemented)
    - Decide which to use and how many of each
- Number of servers to allocate to a single machine (e.g. allocate two conv
  compute to a single machine to hide stalled time waiting for fc to return
  gradients), and also how many GPUs per server (e.g. scheduler may decide
  given a number of machines to create 2 fc compute servers, one per machine,
  or to create 2 but on the same machine, each using 2 GPUs, and use the extra
  machine for another conv compute, or to create a single FC compute/model server
  to minimize communication)
- Data parallelism: Decide how many machines to use within a batch
- How to use each box (CPU, GPU, both, use each GPU as a different server, or use
  4 GPU model/data parallelism, see also below)
- Model parallelism: Decide how many machines to use to partition models,
  or how many GPUs on a single machine (model parallelism across 
  machines unimplemented)
- Precision (lossy, lossless, e.g. given AVX2, lossy compression makes sense)
- Hogwild vs. model averaging vs. other methods (when using many model servers),
  also must choose isolation levels

Inputs:
- Throughput per box (not simply black box however, since a single machine with
  4 GPUs can be seen as a single box or as 4 slower boxes)
- Network speed


--------------------------------------------------------------------------------
Details:
--------------------------------------------------------------------------------

For example, a machine list could be:

master
node015
node019


The servers will then need the following information in the .cfg file:

conv model server (running on node015)
    name = "Example ConvModelServer on 5555";
    bind = "tcp://*:5555";
    type = "ConvModelServer";
    solver_file = "protos/solver.conv_model_server.prototxt";
    data_binary = "protos/dummy.bin";   // Empty file (no file needed, but prevents automatic binary creation)
    output_model_file = "conv_model.bin";

conv compute server (running on node019)
    name = "Example ConvComputeServer on 5555";
    conv_bind = "tcp://node015:5555";
    fc_bind = "tcp://master:5556";
    type = "ConvComputeServer";
    solver_file = "protos/solver.conv_compute_server.prototxt";
    data_binary = "/home/software/dcct/8_train_NEW.bin";

fc server (running on master)
    name = "Example FCComputeModelServer on 5556";
    bind = "tcp://*:5556";
    type = "FCComputeModelServer";
    solver_file = "protos/solver.fc_model_server.prototxt";
    data_binary = "protos/dummy.bin";   // Empty file (no file needed, but prevents automatic binary creation)
    output_model_file = "fc_model.bin";

"""


# ==============================================================================
# Parse arguments
# ==============================================================================

if len(sys.argv) not in [7,8]:
    # SHADJIS TODO: map_fcc_to_cc is only used if many_fc is set.
    # Eventually there will also be an option to map fcm and cm to the same machine,
    # either by making two separate servers and assigning them to one machine or using
    # an AllModelServer (and then also an AllComputeServer can be used)
    # SHADJIS TODO: These can be in a config file if there are too many params
    # SHADJIS TODO: Or even better, rather than a bool for every case (fcc + cc on same machine,
    # etc.), for now we can read in a machine list file which has cols for each machine, e.g.
    # if we have:
    #       master   cc0.0  fcc0
    #       node001  cc0.1
    #       node002  cc1.0  fcc1
    #       node003  cc1.1
    #       node003  fcm
    #       node003  cm
    # then it is clear how machines should be assigned. The scheduler will then eventually
    # generate this file (e.g. a JSON format which specifies each server, its machine,
    # its GPUs, etc.)
    print 'Usage: >>> python run.py  path/to/solver.prototxt  path/to/machine_list.txt  machines_per_batch  CPU|1GPU|4GPU  single_fc|many_fc  (map_fcc_to_cc=)1|0'
    sys.exit(0)

# Check that the distributed cct binary exists before running this script
if not os.path.exists('./dcct'):
    print 'Please first run \'make -j\''
    sys.exit(0)
# We will also need one of the util files, so check here if it exists
if not os.path.exists('./tools/size_util/size_util'):
    print 'Please cd into tools/size_util and \'make -j\''
    sys.exit(0)

solver_file = sys.argv[1]
machine_list_file = sys.argv[2]
machines_per_batch = int(sys.argv[3])       # SHADJIS TODO: Eventually optimizer will select this
node_hw = sys.argv[4]
if sys.argv[5] == 'single_fc':
    single_FC_server = True
    del num_fcc_per_machine     # These are unused
    del num_gpu_per_fcc_machine
else:
    assert sys.argv[5] == 'many_fc'
    print 'Using 1 fc compute server per conv compute group'
    single_FC_server = False
    if sys.argv[6] == '1':
        # If we want to map the fcc to the same machine as cc we need to replace the 
        # port with a local port.
        map_fcc_to_cc = True
        print 'Assigning fcc servers to same machine as (some) cc servers'
    else:
        assert sys.argv[6] == '0'
        map_fcc_to_cc = False

if len(sys.argv) == 8 and sys.argv[7] == 's':
    skip_lmdb_generation = True

if node_hw == 'CPU':
    use_4_gpu = False
    use_1_gpu = False
elif node_hw == '1GPU':
    use_4_gpu = False
    use_1_gpu = True
elif node_hw == '4GPU':
    use_4_gpu = True
    use_1_gpu = False
else:
    assert False

# assert skip_lmdb_generation


# ==============================================================================
# Create input protobuf files (solver and train_val) for each server
# ==============================================================================

# ------------------------------------------------------------------------------
# Read in a machine list so we know how many conv compute servers to create
# ------------------------------------------------------------------------------
# Note: For now, the first line must be the current machine, so there must be at least 1 machine
machine_list = []
f = open(machine_list_file)
for line in f:
    if line.strip():
        machine_list.append(line.strip())
f.close()
assert len(machine_list) >= 1

# Allocate each server to machines

# This depends on how many FC servers we have

if single_FC_server:

    # The FC server is always the first machine
    fc_server_machine = machine_list[0]

    # Conv Model Server:
    # If there are 3 or more machines, assign conv model server to machine 1 (i.e. 2nd one)
    # If there are only 1 or 2 machines, assign this to the first machine (machine 0).
    # That way if there are 2 machines, the conv compute server will have its own machine.
    # (Edit: see comment below, this needs to be fixed)
    if len(machine_list) > 2:
        conv_model_server_machine = machine_list[1]
    else:
        conv_model_server_machine = machine_list[0]

    # Conv Compute Server:
    if len(machine_list) == 1:
        conv_compute_server_machines = [machine_list[0]]
    # SHADJIS TODO: This is wrong, if there are 2 machines it does not help to 
    # have a single conv compute on a separate machine, since it will be idle
    # while fc is running anyway. If 2 machines, we should put conv model and
    # fc on one machine, and conv compute on each machine.
    elif len(machine_list) == 2:
        conv_compute_server_machines = [machine_list[1]]
        num_conv_compute_servers = 1
    else:
        # Calculate the number of conv compute servers
        num_machines_left = len(machine_list)-2
        # Make sure it divides the number of machines per batch
        num_conv_compute_servers = (num_machines_left/machines_per_batch) * machines_per_batch
        conv_compute_server_machines = machine_list[2:2+num_conv_compute_servers]

    # Now determine the number of groups
    num_groups = num_conv_compute_servers / machines_per_batch
    assert num_groups > 0
    assert num_conv_compute_servers % machines_per_batch == 0

else:

    # Determine how to allocate machines
    num_machines = len(machine_list)
    
    # For now let's assert 4 or more machines:
    #  1. On a single machine it doesn't make sense to have multiple fc compute servers, since
    #     they cannot run at the same time so might as well just use a single one
    #  2. SHADJIS TODO: In the case of 2 machines, each machine can have a conv compute and an FC, 
    #     so that data does not need to be copied. However since there is only a single conv model
    #     and fc model server, we need to allocate these. The scheduler will do it later so for
    #     now I will just exit in this case
    #  3. SHADJIS TODO: In the case of 3 machines, it is again possible to have intelligent machine
    #     allocation, but for now I will ignore this case as well
    assert num_machines > 3
    
    # Above are special-cases. It may be beneficial to have multiple fc compute servers when the
    # number of machines < 4, but for now we will assume 4 or more machines and that if there are
    # fewer machines then there will be a single fc compute and model server (since it is less
    # likely to be the bottleneck if there are few conv compute machines)
    
    # For now, if there are 4 or more machines, and FC model / computation are on different machines, 
    # then currently machine allocations will be: 1 conv model, 1 fc model, and a number of conv compute
    # and fc compute specified at the top of the file.
    # In the future it will be possible to have other configurations, e.g. multiple groups allocated
    # to one fc compute server, but for now 1 group per fc will be used.
    
    # The FC model server can be the first machine, and conv model on second
    fc_model_server_machine = machine_list[0]
    conv_model_server_machine = machine_list[1]
    num_machines_left = num_machines - 2
    
    # If num_gpu_per_fcc > 1, then that fcc server will use model 
    # parallelism iff multi_gpu_model_parallelism = True
    num_gpu_per_fcc = num_gpu_per_fcc_machine / num_fcc_per_machine

    # Now assign machines to the fcc and conv compute servers
    # Special case: fcc and cc on same server
    if map_fcc_to_cc:
        num_groups = num_machines_left / machines_per_batch
        num_fc_compute_servers = num_groups
        assert num_groups > 0
        # These FC compute servers will be on the same machines and cc servers, so assign those first
        conv_compute_server_machines = machine_list[2 : 2 + num_groups*machines_per_batch]
        # Now assign FC as a subset of these machines. If group size is 1, it will be the same
        # Otherwise, it will be strided by machines_per_batch
        fc_compute_server_machines = conv_compute_server_machines[::machines_per_batch]
        num_conv_compute_servers = len(conv_compute_server_machines)
    # Default case
    else:
        # Now we have the following:
        #   - num_machines_left (after allocating 1 to conv model and 1 to fc model)
        #   - machines_per_batch (i.e. # machines / conv compute group),
        #   - num fc compute / machine, 
        # Now we can calculate how many parallel batches (AKA groups) there will be:
        #    num_groups = num_machines_left / (num_machines/group)
        #               = num_machines_left / (num_machines/cc_group + num_machines/fcc_server),
        #  where #machines/fcc_server = 1/num_fcc_per_machine <= 1
        num_groups = int( num_machines_left / (machines_per_batch + 1./num_fcc_per_machine) )
        
        # Next we must allocate a number of fc compute servers equal to the number of groups.
        num_fc_compute_servers = num_groups
        assert num_groups > 0
        num_machines_for_fc_compute_servers = ( num_fc_compute_servers + num_fcc_per_machine - 1 )/ num_fcc_per_machine  # Round up
        
        # Allocate machines for these fc compute servers
        fc_compute_server_machines = []
        current_machine = 2 # Since we allocated the first 2 to the model servers
        servers_on_current_machine = 0
        for i in range(num_fc_compute_servers):
            fc_compute_server_machines.append(machine_list[current_machine])
            servers_on_current_machine += 1
            if servers_on_current_machine == num_fcc_per_machine:
                servers_on_current_machine = 0
                current_machine += 1
        # Make sure we assigned all the machines that we had allocated for fcc servers (maybe the last one is
        # not 100% full so check for that case as well)
        assert (current_machine == 2 + num_machines_for_fc_compute_servers) or (current_machine == 2 + num_machines_for_fc_compute_servers - 1)
        current_machine = 2 + num_machines_for_fc_compute_servers
        
        # Now, the remaining number of machines must be able to fit the conv compute servers
        num_machines_for_conv_compute_servers = num_groups * machines_per_batch
        if num_machines_for_conv_compute_servers + current_machine > num_machines:
            print 'Error: your configuration requires more machines than provided (' + \
                str(num_machines) + ' provided, ' + str(num_machines_for_conv_compute_servers + current_machine) + ' needed)'
            assert False
        
        conv_compute_server_machines = machine_list[current_machine : current_machine + num_machines_for_conv_compute_servers]
        num_conv_compute_servers = len(conv_compute_server_machines)

    # Now we have the following set:
    #   - num_groups
    #   - fc_model_server_machine   (as opposed to fc_server_machine for the case of single fc server)
    #   - conv_model_server_machine
    #   - fc_compute_server_machines, num_fc_compute_servers, num_gpu_per_fcc
    #   - conv_compute_server_machines and num_conv_compute_servers, and use_1_gpu or use_4_gpu
    # The rest of the file will use these variables

print 'Number of machines = ' + str(len(machine_list))
print 'Number of groups   = ' + str(num_groups)


# ------------------------------------------------------------------------------
# Find ports for the machines above
# ------------------------------------------------------------------------------
# Now we need to create 2 ports per group (one to listen and one to broadcast)
# SHADJIS TODO: We should get a list of free ports both on the conv model server
# machine and the fc server machine. For now, I am only getting a list of free
# ports on the current machine (running this script) and re-using for the machines
# running these servers, which is wrong. If this causes errors, can log in to 
# each machine and run a script (get_n_free_ports.py), then use those ports.

import socket

# SHADJIS TODO: ssh into conv model / fc server machines and run this there

# We need at least num_groups * 2 unique ports. In fact, this is enough because we can
# re-use these ports #s across machines (to be sure, we should ssh into each and generate
# the ports). But if we ever want to merge the model servers into one machine we need
# more ports. For now we will just re-use the ports.

ports = []
for i in range(num_groups*2):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr = s.getsockname()
    # E.g. parse 40582 from ('0.0.0.0', 40582) SHADJIS TODO: Support other formats
    ports.append(str(addr[1]))
    s.close()
    

# ------------------------------------------------------------------------------
# Parse the solver prototxt to get the name of the train prototxt file
# ------------------------------------------------------------------------------
# From this we need to parse the network prototxt so we can create new 
# sub-networks. We also need to store the solver as a string so we can 
# create new copies of the solver that point to each network prototxt.
f = open(solver_file)
solver_file_lines = []
train_proto = None
for line in f:
    # The line should look something like:
    #   net: "protos/train_test.prototxt"
    match = re.search(r'net\s*:\s*"\s*(\S+)\s*"', line, flags=re.IGNORECASE)
    if match:
        train_proto = match.group(1)
        solver_file_lines.append(line.replace(train_proto, '__TRAIN_PROTO__'))
    else:
        solver_file_lines.append(line)
assert train_proto
solver_file_str = ''.join(solver_file_lines)
f.close()


# ------------------------------------------------------------------------------
# Parse the train prototxt file for lmdb names
# ------------------------------------------------------------------------------
# First parse the file to obtain the lmdb names
# Also while parsing, count the number of FC layers. This is only used for 
# multi-GPU model parallelism
num_fc_layers_total = 0
f = open(train_proto)
lmdb_databases = []
for line in f:
    # E.g. a line like
    #  source: "/home/ubuntu/train_data"
    match = re.search(r'source\s*:\s*"\s*(\S+)\s*"', line, flags=re.IGNORECASE)
    if match:
        lmdb_databases.append(match.group(1))
    
    # Also check if this is an fc layer
    match = re.search(r'type\s*:\s*"\s*(\S+)\s*"', line, flags=re.IGNORECASE)
    if match:
        type = match.group(1)
        if 'INNERPRODUCT' in type.upper():
            num_fc_layers_total += 1

assert len(lmdb_databases) in [1,2]
assert num_fc_layers_total > 0    # For softmax
if len(lmdb_databases) == 2:
    print 'Warning: For now, validation / test sets are ignored.'
    print '         This will be supported soon.'
f.close()

# SHADJIS TODO: Support validation set lmdb
train_lmdb_name = lmdb_databases[0].rstrip('/')
conv_movel_server_train_lmdb_name = train_lmdb_name
fc_server_train_lmdb_name = train_lmdb_name + '_FC'
conv_compute_server_train_lmdb_names = []
for i in range(num_conv_compute_servers):
    conv_compute_server_train_lmdb_names.append(train_lmdb_name + '_p' + str(i))


# ------------------------------------------------------------------------------
# Parse the network train prototxt again, now to split into 2 networks
# ------------------------------------------------------------------------------
# Now read through the file again, this time partitioning into conv and fc layers
# Here we want to make 2 networks: first with everything up to first FC, then
# first FC to the end
# Note: in the case of separate fcc and fcm servers, the protos are identical
# except GPU info. However, the GPU info can be complicated because fcm has no
# GPU (this is easy to handle) but fcc's potentially each have different GPUs.
conv_model_server_proto_str = ''
conv_compute_server_proto_strs = ['']*num_conv_compute_servers
if single_FC_server:
    fc_server_proto_str = ''
else:
    fcm_server_proto_str = ''
    fcc_server_proto_strs = ['']*num_fc_compute_servers

# Where we are in the file
data_section = True
conv_section = False
fc_section = False

# Read layer by layer
lines_for_current_layer = []
lines_for_current_layer_no_gpu = []
f = open(train_proto)
num_fc_layers_found = 0
for line in f:
    # Check if this is the start of a new layer
    # SHADJIS TODO: I think proto can skip a line before the curly brace but I'll ignore that for now
    if re.search(r'layer\s*\{', line, flags=re.IGNORECASE):
        layer_str = ''.join(lines_for_current_layer)
        layer_str_no_gpu = ''.join(lines_for_current_layer_no_gpu)
        lines_for_current_layer = [line]
        lines_for_current_layer_no_gpu = [line]
        # This is a new layer. What we do with the old layer depends on 
        # which section we are in
        
        # Case 1, this is a data layer
        if data_section:
            # We want to append this layer to all the networks
            conv_model_server_proto_str += layer_str_no_gpu # No GPU for now on the conv model server. SHADJIS TODO: Why is this needed for data section? Assert same as no gpu
            # For each conv compute network, we need to replace the LMDB
            # We also need to reduce the batch size
            layer_str_copy = layer_str
            match = re.search(r'batch_size\s*:\s*(\d+)', layer_str, flags=re.IGNORECASE)
            if match:
                batch_size = int(match.group(1))
                assert batch_size % machines_per_batch == 0
                batch_size_reduced = int(batch_size / machines_per_batch)
                layer_str_copy = re.sub(r'batch_size\s*:\s*(\d+)', 'batch_size: ' + str(batch_size_reduced), layer_str_copy, 1, flags=re.IGNORECASE)
            for i in range(num_conv_compute_servers):
                conv_compute_server_proto_strs[i] += layer_str_copy.replace(conv_movel_server_train_lmdb_name, conv_compute_server_train_lmdb_names[i])
            # For the FC network, we need to do some replacements and then also replace the LMDB:
            # SHADJIS TODO: Use regex to replace the mirror in mirror: true/false, but not others
            layer_str = re.sub(r'mirror',    '#mirror',    layer_str , 1, flags=re.IGNORECASE)
            layer_str = re.sub(r'crop_size', '#crop_size', layer_str , 1, flags=re.IGNORECASE)
            layer_str = re.sub(r'mean_file', '#mean_file', layer_str , 1, flags=re.IGNORECASE)
            layer_str_for_fc = layer_str.replace(conv_movel_server_train_lmdb_name, fc_server_train_lmdb_name)
            if single_FC_server:
                fc_server_proto_str += layer_str_for_fc
            else:
                fcm_server_proto_str += layer_str_for_fc
                for i in range(num_fc_compute_servers):
                    fcc_server_proto_strs[i] += layer_str_for_fc
        # Case 2, this is a layer in the conv part
        elif conv_section:
            conv_model_server_proto_str += layer_str_no_gpu
            for i in range(num_conv_compute_servers):
                conv_compute_server_proto_strs[i] += layer_str
        # Case 3, this is a layer in the FC part
        elif fc_section:
            if single_FC_server:
                fc_server_proto_str += layer_str
            else:
                fcm_server_proto_str += layer_str_no_gpu
                # Now we need to substitute in {SUB_GPU_NUM_HERE} for the correct gpu numbers
                # Iterate over each server and assign GPUs. Keep in mind that multiple servers
                # may be assigned to a single machine
                # Start at machine 0 GPU 0 and increment the GPU each time. Reset once we 
                # reach num_gpu_per_fcc_machine (this moves to a new machine, although the
                # machine assignments are not done here, since they were already done above)
                current_gpu = 0
                for i in range(num_fc_compute_servers):
                    layer_str_this_fcc_server = layer_str
                    for g in range(num_gpu_per_fcc):
                        layer_str_this_fcc_server = layer_str_this_fcc_server.replace('{SUB_GPU_NUM_HERE}', str(current_gpu), 1)
                        current_gpu += 1
                        if current_gpu == num_gpu_per_fcc_machine:
                            current_gpu = 0
                    fcc_server_proto_strs[i] += layer_str_this_fcc_server
            
    # Otherwise this is part of a layer
    else:
        lines_for_current_layer.append(line)
        lines_for_current_layer_no_gpu.append(line)
        
        # We can also determine if we moved to a new section of the network
        match = re.search(r'type\s*:\s*"\s*(\S+)\s*"', line, flags=re.IGNORECASE)
        if match:
            # If this is a 'type: "..."' line, the section of the network might have changed
            type = match.group(1)
            # If it is a convolution layer, assert we are not in the fc section
            # and transition to conv section if in data section
            if 'CONVOLUTION' in type.upper() or 'POOL' in type.upper():
                assert not fc_section
                if data_section:
                    data_section = False
                    conv_section = True
            elif 'INNERPRODUCT' in type.upper():
                num_fc_layers_found += 1
                data_section = False
                conv_section = False
                fc_section = True
                
            # Update proto with GPU information
            #
            # Only do this once per layer, i.e. we want to do it after this 'type:' line:
            #
            #   type: "ReLU"            <----
            #
            # But not after this 'type:' line
            #
            #    weight_filler {
            #      type: "gaussian"     <----
            #      std: 0.01
            #    }
            if type.upper() in ['INNERPRODUCT', 'RELU', 'DROPOUT', 'POOLING', 'CONVOLUTION', 'LRN']:
            
                # Conv can use up to 4 GPUs
                if conv_section:
                    if use_4_gpu:
                        lines_for_current_layer.append('''  gpu_0_batch_proportion: 0.25
  gpu_1_batch_proportion: 0.25
  gpu_2_batch_proportion: 0.25
  gpu_3_batch_proportion: 0.25
''')
                    elif use_1_gpu:
                        lines_for_current_layer.append('''  gpu_0_batch_proportion: 1.0
''')
                # FC can use up to 1 GPU with model parallelism disabled,
                # and all the GPUs with model parallelism enabled
                elif fc_section and type.upper() != 'SOFTMAXWITHLOSS':
                    if single_FC_server:
                        if use_1_gpu or (use_4_gpu and not multi_gpu_model_parallelism):
                            lines_for_current_layer.append('''  gpu_0_batch_proportion: 1.0
''')
                        # If 4 GPUs and model parallelism is enabled, the type of parallelism and
                        # number of GPUs depends on the layer. 
                        # By default, FC layers get 4 GPUs and model parallelism, and non-FC layers
                        # get 4 GPUs and data parallelism. However the final FC layer may be faster
                        # on just 1 GPU because:
                        #  - 4 GPUs data parallelism: this is slow because the computation is small
                        #    for the final FC layer so there is little speedup  from using
                        #    multiple GPUs, but accumulating the gradients requires
                        #    copying the gradients from each GPU and model back to each GPU
                        #  - 4 GPUs model parallelism: this is fast but requires copying all the
                        #    data to each GPU then back to the host in backward pass to sum gradients
                        # It turns out that because the computation is fast for the last FC, minimizing copies 
                        # is more important, so using 1 GPU (and keeping gradients on device at all times)
                        # is fastest. Then to avoid copies to that GPU, it is also fastest if all
                        # the preceding layers (e.g. ReLU, dropout) also use 1 GPU (1 GPU is batch
                        # parallelism by default, but 1 GPU batch and 1 GPU depth are equivalent).
                        elif use_4_gpu:
                            # Now how many GPUs to use depends on the layer
                            assert multi_gpu_model_parallelism
                            # If we are past the second-last FC, use 1 GPU
                            # Note that this conditional branch seems useless (same result in both cases)
                            # but is not because of batch vs depth proportion
                            if type.upper() == 'INNERPRODUCT':
                                assert num_fc_layers_found >= 1
                                assert num_fc_layers_found <= num_fc_layers_total
                                if num_fc_layers_found == num_fc_layers_total:  # Last FC
                                    lines_for_current_layer.append('''  gpu_0_batch_proportion: 1.0
''')
                                else:
                                    lines_for_current_layer.append('''  gpu_0_depth_proportion: 0.25
  gpu_1_depth_proportion: 0.25
  gpu_2_depth_proportion: 0.25
  gpu_3_depth_proportion: 0.25
''')
                            else:
                                if num_fc_layers_found >= num_fc_layers_total-1:  # Right before last FC
                                    lines_for_current_layer.append('''  gpu_0_batch_proportion: 1.0
''')
                                else:
                                    lines_for_current_layer.append('''  gpu_0_batch_proportion: 0.25
  gpu_1_batch_proportion: 0.25
  gpu_2_batch_proportion: 0.25
  gpu_3_batch_proportion: 0.25
''')
                    # not single_FC_server
                    else:
                        if num_gpu_per_fcc == 1 or (num_gpu_per_fcc > 1 and not multi_gpu_model_parallelism):
                            lines_for_current_layer.append('''  gpu_{SUB_GPU_NUM_HERE}_batch_proportion: 1.0
''')
                        # See model parallelism comment in code above
                        elif num_gpu_per_fcc > 1:
                            # Now how many GPUs to use depends on the layer
                            assert multi_gpu_model_parallelism
                            # If we are past the second-last FC, use 1 GPU
                            if type.upper() == 'INNERPRODUCT':
                                assert num_fc_layers_found >= 1
                                assert num_fc_layers_found <= num_fc_layers_total
                                if num_fc_layers_found == num_fc_layers_total:  # Last FC
                                    lines_for_current_layer.append('''  gpu_{SUB_GPU_NUM_HERE}_batch_proportion: 1.0
''')
                                elif num_gpu_per_fcc == 2:
                                    lines_for_current_layer.append('''  gpu_{SUB_GPU_NUM_HERE}_depth_proportion: 0.5
  gpu_{SUB_GPU_NUM_HERE}_depth_proportion: 0.5
''')
                                else:
                                    assert num_gpu_per_fcc == 4
                                    lines_for_current_layer.append('''  gpu_{SUB_GPU_NUM_HERE}_depth_proportion: 0.25
  gpu_{SUB_GPU_NUM_HERE}_depth_proportion: 0.25
  gpu_{SUB_GPU_NUM_HERE}_depth_proportion: 0.25
  gpu_{SUB_GPU_NUM_HERE}_depth_proportion: 0.25
''')
                            else:
                                if num_fc_layers_found >= num_fc_layers_total-1:  # Right before last FC
                                    lines_for_current_layer.append('''  gpu_{SUB_GPU_NUM_HERE}_batch_proportion: 1.0
''')
                                elif num_gpu_per_fcc == 2:
                                    lines_for_current_layer.append('''  gpu_{SUB_GPU_NUM_HERE}_batch_proportion: 0.5
  gpu_{SUB_GPU_NUM_HERE}_batch_proportion: 0.5
''')
                                else:
                                    assert num_gpu_per_fcc == 4
                                    lines_for_current_layer.append('''  gpu_{SUB_GPU_NUM_HERE}_batch_proportion: 0.25
  gpu_{SUB_GPU_NUM_HERE}_batch_proportion: 0.25
  gpu_{SUB_GPU_NUM_HERE}_batch_proportion: 0.25
  gpu_{SUB_GPU_NUM_HERE}_batch_proportion: 0.25
''')


f.close()

# Call code above for the last layer too now
layer_str = ''.join(lines_for_current_layer)
layer_str_no_gpu = ''.join(lines_for_current_layer_no_gpu)
assert layer_str == layer_str_no_gpu # Last layer (softmax) should not use GPU
assert fc_section
if single_FC_server:
    fc_server_proto_str += layer_str
else:
    fcm_server_proto_str += layer_str_no_gpu
    for i in range(num_fc_compute_servers):
        fcc_server_proto_strs[i] += layer_str


# ------------------------------------------------------------------------------
# Create new solver and network prototxt files
# ------------------------------------------------------------------------------
input_file_dir = 'server_input_files-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
os.system('mkdir -p ' + input_file_dir)
dummy_file = input_file_dir + '/dummy.bin'
os.system('touch ' + dummy_file)
print 'Writing input prototxt files to ' + input_file_dir + '/'

# First create the solver files for the model servers
if single_FC_server:
    server_types = ['conv_model', 'fc_model_compute']
else:
    server_types = ['conv_model', 'fc_model']
for server in server_types:

    solver_name = input_file_dir + '/solver.' + server + '_server.prototxt'
    train_name  = input_file_dir + '/train_val.' + server + '_server.prototxt'
    
    # Make a solver for this server
    f = open(solver_name, 'w')
    print '  Writing ' + solver_name
    f.write(solver_file_str.replace('__TRAIN_PROTO__', train_name))
    f.close()
    
    # Make a train file for this server
    f = open(train_name, 'w')
    print '  Writing ' + train_name
    if server == 'conv_model':
        f.write(conv_model_server_proto_str)
        conv_model_server_solver_file = solver_name
    elif server == 'fc_model_compute':
        assert single_FC_server
        f.write(fc_server_proto_str)
        fc_server_solver_file = solver_name
    else:
        assert server == 'fc_model'
        assert not single_FC_server
        f.write(fcm_server_proto_str)
        fcm_server_solver_file = solver_name
    f.close()
    
# Now create the solver files for the conv compute servers
conv_compute_server_solver_files = []
for i in range(num_conv_compute_servers):

    solver_name = input_file_dir + '/solver.conv_compute_server.' + str(i) + '.prototxt'
    conv_compute_server_solver_files.append(solver_name)
    train_name  = input_file_dir + '/train_val.conv_compute_server.' + str(i) + '.prototxt'
    
    # Make a solver for this server
    f = open(solver_name, 'w')
    print '  Writing ' + solver_name
    f.write(solver_file_str.replace('__TRAIN_PROTO__', train_name))
    f.close()

    # Make a train file for this server
    f = open(train_name, 'w')
    print '  Writing ' + train_name
    f.write(conv_compute_server_proto_strs[i])
    f.close()

# Now create the solver files for the fc compute servers, if any
if not single_FC_server:
    fc_compute_server_solver_files = []
    for i in range(num_fc_compute_servers):

        solver_name = input_file_dir + '/solver.fc_compute_server.' + str(i) + '.prototxt'
        fc_compute_server_solver_files.append(solver_name)
        train_name  = input_file_dir + '/train_val.fc_compute_server.' + str(i) + '.prototxt'
        
        # Make a solver for this server
        f = open(solver_name, 'w')
        print '  Writing ' + solver_name
        f.write(solver_file_str.replace('__TRAIN_PROTO__', train_name))
        f.close()

        # Make a train file for this server
        f = open(train_name, 'w')
        print '  Writing ' + train_name
        f.write(fcc_server_proto_strs[i])
        f.close()


# ==============================================================================
# Create the LMDBs referenced above
# ==============================================================================

# Recall that above we made:
#
#   conv_movel_server_train_lmdb_name      (same as default, since unused)
#   fc_server_train_lmdb_name
# and
#   conv_compute_server_train_lmdb_names for i in range(num_conv_compute_servers)
#
# Now we need to create the fc and conv compute lmdb files by reading in the
# conv model server. The fc server can also be empty but needs the right size.
# We don't know this size however without loading the network, so we need to
# do that using a C++ utility (see below)


# ------------------------------------------------------------------------------
# First, make the lmdb for each conv compute server
# ------------------------------------------------------------------------------

if not skip_lmdb_generation:

    # SHADJIS TODO: Skip this if there is only 1 partition, and re-use existing lmdb

    # Open the full lmdb that we will read from
    read_lmdb_name = conv_movel_server_train_lmdb_name
    map_size = 1024*1024*1024*1024

    # First count the number of images
    read_env = lmdb.open(read_lmdb_name, readonly=True)
    num_images = 0
    with read_env.begin() as read_txn:
        with read_txn.cursor() as read_cursor:
            for key, value in read_cursor:
                num_images += 1
    read_env.close()
    print 'LMDB ' + read_lmdb_name + ' contains ' + str(num_images)

    # Now split by the number of conv compute servers
    num_images_per_conv_compute_server = [num_images/num_conv_compute_servers]*num_conv_compute_servers

    # We also have to add the remainders
    num_leftover = num_images%num_conv_compute_servers
    for i in range(num_leftover):
        num_images_per_conv_compute_server[i] += 1
    assert sum(num_images_per_conv_compute_server) == num_images

    # Now create the lmdb for each conv compute server

    def open_new_write_lmdb_helper(new_lmdb_name, num_imgs, map_size):
        os.system('rm -rf ' + new_lmdb_name)
        os.system('rm -rf ' + new_lmdb_name + '.bin')
        write_env = lmdb.open(new_lmdb_name, readonly=False, lock=False, map_size=map_size)
        write_txn = write_env.begin(write=True)
        print '  Writing ' + str(num_imgs) + ' images to ' + new_lmdb_name
        return write_env, write_txn

    read_env = lmdb.open(read_lmdb_name, readonly=True)
    with read_env.begin() as read_txn:
        with read_txn.cursor() as read_cursor:
        
            # Read over each datum again, this time writing to a new lmdb
            current_server = 0
            img_idx = 0
            # Open the LMDB for the first server
            write_env, write_txn = open_new_write_lmdb_helper(conv_compute_server_train_lmdb_names[current_server], num_images_per_conv_compute_server[current_server], map_size)
            
            # Read over each image in original lmdb
            for key, value in read_cursor:
            
                # Check if we should move to the next server
                if img_idx >= num_images_per_conv_compute_server[current_server]:
                    # We just finished the current server
                    # First close the currenet lmdb
                    write_txn.commit()
                    write_env.close()
                    # Increment server count and reset img_idx
                    current_server += 1
                    img_idx = 0
                    # Open new lmdb
                    write_env, write_txn = open_new_write_lmdb_helper(conv_compute_server_train_lmdb_names[current_server], num_images_per_conv_compute_server[current_server], map_size)
                
                # Write the new datum to the new lmdb
                write_txn.put(key, value)
                img_idx += 1

            # assert we have 1 server lmdb left to write
            assert current_server == num_conv_compute_servers-1
            assert img_idx == num_images_per_conv_compute_server[current_server]
            write_txn.commit()
            write_env.close()
            
    read_env.close()


# ------------------------------------------------------------------------------
# Next, make the lmdb for the FC server
# ------------------------------------------------------------------------------

if not skip_lmdb_generation:

    # This requires calling a utility which loads the network and prints the size
    # of the output of the final conv layer.
    util_output_str = subprocess.check_output(['./tools/size_util/size_util', conv_model_server_solver_file, dummy_file])
    num_fc_inputs = int(util_output_str.strip().split("\n")[-1].strip())

    # Now create a new LMDB with 1 datum that contains the right size
    write_env, write_txn = open_new_write_lmdb_helper(fc_server_train_lmdb_name, 1, map_size)

    # Create the datum
    datum = datum_pb2.Datum()
    datum.height = 1
    datum.width  = 1
    datum.channels = num_fc_inputs

    # Write back the new datum
    write_txn.put('dummy', datum.SerializeToString())

    write_txn.commit()
    write_env.close()


# ==============================================================================
# Create config files for each server
# ==============================================================================

print 'Generating configuration files for each server'

# Also keep a list of the machine and config file
cmd_params = []

# Get the config file names
if single_FC_server:
    fc_server_cfg = input_file_dir + '/fc_server.cfg'
else:
    fcm_server_cfg = input_file_dir + '/fc_model_server.cfg'
    fcc_server_cfgs = []
    for i in range(num_fc_compute_servers):
        fcc_server_cfgs.append(input_file_dir + '/fc_compute_server.' + str(i) + '.cfg')
conv_model_server_cfg = input_file_dir + '/conv_model_server.cfg'
conv_compute_server_cfgs = []
for i in range(num_conv_compute_servers):
    conv_compute_server_cfgs.append(input_file_dir + '/conv_compute_server.' + str(i) + '.cfg')

# Write config files

if single_FC_server:
    # FC server
    print '  Writing ' + fc_server_cfg
    f = open(fc_server_cfg, 'w')
    f.write('''name = "FCComputeModelServer on tcp://''' + fc_server_machine + '''";
type = "FCComputeModelServer";
solver = "''' + fc_server_solver_file + '''";
train_bin = "''' + dummy_file + '''";
group_size = ''' + str(machines_per_batch) + ''';

ports = (
''')
    for i in range(num_groups):
        if i != 0:
            f.write(',')
        f.write('''
  {
    broadcast = "tcp://*:''' + str(ports[2*i    ]) + '''",
    listen = "tcp://*:'''    + str(ports[2*i + 1]) + '''"
  }''')
    f.write('''
);
''')
    f.close()
    cmd_params.append((fc_server_machine, fc_server_cfg))
else:
    # FC model server
    # Note group_size = 1 since the fc compute servers will be connecting to this
    print '  Writing ' + fcm_server_cfg
    f = open(fcm_server_cfg, 'w')
    f.write('''name = "FCModelServer on tcp://''' + fc_model_server_machine + '''";
type = "FCModelServer";
solver = "''' + fcm_server_solver_file + '''";
train_bin = "''' + dummy_file + '''";
group_size = 1;

ports = (
''')
    for i in range(num_groups):
        if i != 0:
            f.write(',')
        f.write('''
  {
    broadcast = "tcp://*:''' + str(ports[2*i    ]) + '''",
    listen = "tcp://*:'''    + str(ports[2*i + 1]) + '''"
  }''')
    f.write('''
);
''')
    f.close()
    cmd_params.append((fc_model_server_machine, fcm_server_cfg))

# Conv model server
print '  Writing ' + conv_model_server_cfg
f = open(conv_model_server_cfg, 'w')
f.write('''name = "ConvModelServer on tcp://''' + conv_model_server_machine + '''";
type = "ConvModelServer";
solver = "''' + conv_model_server_solver_file + '''";
train_bin = "''' + dummy_file + '''";
group_size = ''' + str(machines_per_batch) + ''';

ports = (
''')
for i in range(num_groups):
    if i != 0:
        f.write(',')
    f.write('''
  {
    broadcast = "tcp://*:''' + str(ports[2*i    ]) + '''",
    listen = "tcp://*:'''    + str(ports[2*i + 1]) + '''"
  }''')
f.write('''
);
''')
f.close()
cmd_params.append((conv_model_server_machine, conv_model_server_cfg))

# Conv compute servers
for i in range(num_conv_compute_servers):
    group_of_this_machine = i / machines_per_batch
    print '  Writing ' + conv_compute_server_cfgs[i]
    f = open(conv_compute_server_cfgs[i], 'w')
    
    if single_FC_server:
        fc_bind_machine = fc_server_machine
    else:
        # Special case: if these are on the same machine, use localhost
        if conv_compute_server_machines[i] == fc_compute_server_machines[group_of_this_machine]:
            fc_bind_machine = '127.0.0.1'   # Localhost
        else:
            fc_bind_machine = fc_compute_server_machines[group_of_this_machine]
    
    f.write('''name = "ConvComputeServer ''' + str(i) + '''";
conv_listen_bind = "tcp://''' + conv_model_server_machine + ''':''' + str(ports[2*group_of_this_machine + 1]) + '''";
conv_send_bind = "tcp://'''   + conv_model_server_machine + ''':''' + str(ports[2*group_of_this_machine    ]) + '''";
fc_listen_bind = "tcp://'''   + fc_bind_machine           + ''':''' + str(ports[2*group_of_this_machine + 1]) + '''";
fc_send_bind = "tcp://'''     + fc_bind_machine           + ''':''' + str(ports[2*group_of_this_machine    ]) + '''";
type = "ConvComputeServer";
solver = "''' + conv_compute_server_solver_files[i] + '''";
train_bin = "''' + conv_compute_server_train_lmdb_names[i] + '''.bin";
group_size = ''' + str(machines_per_batch) + ''';
rank_in_group = ''' + str(i%machines_per_batch) + ''';
''')
    f.close()
    cmd_params.append((conv_compute_server_machines[i], conv_compute_server_cfgs[i]))

# FC compute servers
if not single_FC_server:
    # Note rank in group is always 0 because there is only one fcc per group
    for i in range(num_fc_compute_servers):
        print '  Writing ' + fcc_server_cfgs[i]
        f = open(fcc_server_cfgs[i], 'w')
        f.write('''name = "FCComputeServer ''' + str(i) + '''";
conv_listen_bind = "tcp://''' + '*'                     + ''':''' + str(ports[2*i + 1]) + '''";
conv_send_bind = "tcp://'''   + '*'                     + ''':''' + str(ports[2*i    ]) + '''";
fc_listen_bind = "tcp://'''   + fc_model_server_machine + ''':''' + str(ports[2*i + 1]) + '''";
fc_send_bind = "tcp://'''     + fc_model_server_machine + ''':''' + str(ports[2*i    ]) + '''";
type = "FCComputeServer";
solver = "''' + fc_compute_server_solver_files[i] + '''";
train_bin = "''' + dummy_file + '''";
group_size = ''' + str(machines_per_batch) + ''';
rank_in_group = 0;
''')
        f.close()
        cmd_params.append((fc_compute_server_machines[i], fcc_server_cfgs[i]))



# ==============================================================================
# Run ssh commands
# ==============================================================================

print '''
Beginning to run commands for each server (commands also written to rerun_experiment.sh)
'''

# Run the commmands
f = open('rerun_experiment.sh', 'w')
for cmd_param in cmd_params:
    machine  = cmd_param[0]
    cfg_file = cmd_param[1]
    cmd = 'ssh ' + user + '@' + machine + ' \'' + extra_cmd + ' ./dcct ' + cfg_file + ' &> ' + cfg_file +'.out\' &'
    f.write(cmd + "\n")
    print cmd
    os.system(cmd)
    
    # SHADJIS TODO: To prevent FC (ZMQ SUB) from missing model from CM, sleep (make more permanent solution later)
    # Most of the time it works to sleep 15s only after model servers, but then for 32 machines, 1 group it hangs.
    # However if I sleep 0 after each conv compute (i.e. do nothing differently) it works. Since this is a hack 
    # anyway and has to do with e.g. OS scheduling it is probably not that unexpected and probably random.
    if 'fc_server' in cmd or 'conv_model_server' in cmd or 'fc_model_server' in cmd:
        cmd = 'sleep 15'
    elif 'fc_compute_server' in cmd:        # SHADJIS TODO: May be able to reduce this delay
        cmd = 'sleep 5'
    else:
        cmd = 'sleep 1' # sleep 0 works too, but not removing the command entirely
    f.write(cmd + "\n")
    print cmd
    os.system(cmd)
f.close()

# Also generate a script that can be used to kill all these servers
f = open('kill_servers.sh', 'w')
for cmd_param in cmd_params:
    machine  = cmd_param[0]
    # f.write('ssh ' + user + '@' + machine + ' \'pkill dcct; fuser -k 5555/tcp; fuser -k 5556/tcp;\' &' + "\n")
    f.write('ssh ' + user + '@' + machine + ' \'pkill dcct;\' &' + "\n")
f.close()

print '''
Servers are now running on each machine over ssh.
See the input configuration file for each server to see which machine it is running on.
To stop all servers, run:

$ bash kill_servers.sh
'''
