# ==============================================================================
# 
# run.py
# 
# usage:
# 
# $ python run.py path/to/solver.prototxt path/to/machine_list.txt
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

# Extra commands to run following ssh (e.g. cd into right dir)
# Note also that ssh does not source .bashrc so may need to run load commands as well.
extra_cmd = 'cd ../home/software/dcct/; export PATH=$PATH:/usr/local/cuda-7.0/bin; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.0/lib64;'


# ------------------------------------------------------------------------------
# Script parameters
# ------------------------------------------------------------------------------

# Set to true after lmdb has been generated once (saves time)
skip_lmdb_generation = True


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

if len(sys.argv) != 3:
    print 'Usage: >>> python run.py  path/to/solver.prototxt  path/to/machine_list.txt'
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

# The FC server is always the first machine
fc_server_machine = machine_list[0]

# Conv Model Server:
# If there are 3 or more machines, assign conv model server to machine 1 (i.e. 2nd one)
# If there are only 1 or 2 machines, assign this to the first machine (machine 0).
# That way if there are 2 machines, the conv compute server will have its own machine.
if len(machine_list) > 2:
    conv_model_server_machine = machine_list[1]
else:
    conv_model_server_machine = machine_list[0]

# Conv Compute Server:
if len(machine_list) == 1:
    conv_compute_server_machines = [machine_list[0]]
elif len(machine_list) == 2:
    conv_compute_server_machines = [machine_list[1]]
else:
    conv_compute_server_machines = machine_list[2:]
num_conv_compute_servers = len(conv_compute_server_machines)


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
f = open(train_proto)
lmdb_databases = []
for line in f:
    # E.g. a line like
    #  source: "/home/ubuntu/train_data"
    match = re.search(r'source\s*:\s*"\s*(\S+)\s*"', line, flags=re.IGNORECASE)
    if match:
        lmdb_databases.append(match.group(1))
assert len(lmdb_databases) in [1,2]
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
conv_model_server_proto_str = ''
fc_server_proto_str = ''
conv_compute_server_proto_strs = ['']*num_conv_compute_servers

# Where we are in the file
data_section = True
conv_section = False
fc_section = False

# Read layer by layer
lines_for_current_layer = []
f = open(train_proto)
for line in f:
    # Check if this is the start of a new layer
    # SHADJIS TODO: I think proto can skip a line bfore the curly brace but I'll ignore that for now
    if re.search(r'layer\s*\{', line, flags=re.IGNORECASE):
        layer_str = ''.join(lines_for_current_layer)
        lines_for_current_layer = [line]
        # This is a new layer. What we do with the old layer depends on 
        # which section we are in
        
        # Case 1, this is a data layer
        if data_section:
            # We want to append this layer to all the networks
            conv_model_server_proto_str += layer_str
            # For each conv compute network, we need to replace the LMDB
            for i in range(num_conv_compute_servers):
                conv_compute_server_proto_strs[i] += layer_str.replace(conv_movel_server_train_lmdb_name, conv_compute_server_train_lmdb_names[i])
            # For the FC network, we need to do some replacements and then also replace the LMDB:
            layer_str = re.sub(r'mirror',    '#mirror',    layer_str , 1, flags=re.IGNORECASE)
            layer_str = re.sub(r'crop_size', '#crop_size', layer_str , 1, flags=re.IGNORECASE)
            layer_str = re.sub(r'mean_file', '#mean_file', layer_str , 1, flags=re.IGNORECASE)
            fc_server_proto_str += layer_str.replace(conv_movel_server_train_lmdb_name, fc_server_train_lmdb_name)
        # Case 2, this is a layer in the conv part
        elif conv_section:
            conv_model_server_proto_str += layer_str
            for i in range(num_conv_compute_servers):
                conv_compute_server_proto_strs[i] += layer_str
        # Case 3, this is a layer in the FC part
        elif fc_section:
            fc_server_proto_str += layer_str
            
    # Otherwise this is part of a layer
    else:
        lines_for_current_layer.append(line)
        
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
                data_section = False
                conv_section = False
                fc_section = True
f.close()
        
# Call code above for the last layer too now
layer_str = ''.join(lines_for_current_layer)
assert fc_section
fc_server_proto_str += layer_str


# ------------------------------------------------------------------------------
# Create new solver and network prototxt files
# ------------------------------------------------------------------------------
input_file_dir = 'server_input_files-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
os.system('mkdir -p ' + input_file_dir)
dummy_file = input_file_dir + '/dummy.bin'
os.system('touch ' + dummy_file)
print 'Writing input prototxt files to ' + input_file_dir + '/'

# First create the solver files for the model servers
for server in ['conv_model', 'fc_model_compute']:

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
    else:
        assert server == 'fc_model_compute'
        f.write(fc_server_proto_str)
        fc_server_solver_file = solver_name
    f.close()
    

# Now create the solver files for the conv compute servers
conv_compute_server_solver_files = []
for i in range(num_conv_compute_servers):

    solver_name = input_file_dir + '/solver.conv_compute_server.' + str(i) + '.prototxt'
    conv_compute_server_solver_files.append(solver_name)
    train_name  = input_file_dir + '/train_val.conv_compute_server_server.' + str(i) + '.prototxt'
    
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
fc_server_cfg = input_file_dir + '/fc_server.cfg'
conv_model_server_cfg = input_file_dir + '/conv_model_server.cfg'
conv_compute_server_cfgs = []
for i in range(num_conv_compute_servers):
    conv_compute_server_cfgs.append(input_file_dir + '/conv_compute_server.' + str(i) + '.cfg')

# Write config files

# FC server
print '  Writing ' + fc_server_cfg
f = open(fc_server_cfg, 'w')
f.write('''name = "FCComputeModelServer on tcp://''' + fc_server_machine + ''':5556";
bind = "tcp://*:5556";
type = "FCComputeModelServer";
solver = "''' + fc_server_solver_file + '''"
train_bin = "''' + dummy_file + '''"
''')
f.close()
cmd_params.append((fc_server_machine, fc_server_cfg))

# Conv model server
print '  Writing ' + conv_model_server_cfg
f = open(conv_model_server_cfg, 'w')
f.write('''name = "ConvModelServer on tcp://''' + conv_model_server_machine + ''':5555";
bind = "tcp://*:5555";
type = "ConvModelServer";
solver = "''' + conv_model_server_solver_file + '''"
train_bin = "''' + dummy_file + '''"
''')
f.close()
cmd_params.append((conv_model_server_machine, conv_model_server_cfg))

# Conv compute servers
for i in range(num_conv_compute_servers):
    print '  Writing ' + conv_compute_server_cfgs[i]
    f = open(conv_compute_server_cfgs[i], 'w')
    f.write('''name = "ConvComputeServer ''' + str(i) + '''";
conv_bind = "tcp://''' + conv_model_server_machine + ''':5555";
fc_bind = "tcp://''' + fc_server_machine + ''':5556";
type = "ConvComputeServer";
solver = "''' + conv_compute_server_solver_files[i] + '''"
train_bin = "''' + input_file_dir + '/conv_compute_train_data.' + str(i) + '.bin' + '''"
''')
    f.close()
    cmd_params.append((conv_compute_server_machines[i], conv_compute_server_cfgs[i]))


# ==============================================================================
# Run ssh commands
# ==============================================================================

# Run the commmands
for cmd_param in cmd_params:
    machine  = cmd_param[0]
    cfg_file = cmd_param[1]
    cmd = 'ssh ' + user + '@' + machine + ' \'' + extra_cmd + ' ./dcct ' + cfg_file + ' &> ' + cfg_file +'.out\' &'
    print cmd
    os.system(cmd)

