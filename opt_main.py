# ==============================================================================
# 
# opt_main.py
# 
# usage:
# 
# $ python  opt_main.py  config.txt
# 
# ==============================================================================

import os
import sys

final_time = 0
running_avg = 8
random_seeds = [1]  # Seeds matter a lot, but will not search now (save time)
momentum_list_phase_0 = [0.0, 0.3, 0.6, 0.9]
FCC = False
EXTRA_TIME_FIRST = 0
MAKE_LMDB_FIRST = False
#LR_list =


# ==============================================================================
# Description
# ==============================================================================
"""

This reads an input config file and calls run.py for a number of configurations

TODO:
- Once we finish tuning for G groups, and we are about to start for 2G, can
  reduce params to search since we know LR cannot be greater (momentum can
  however if LR goes down)
- Change display frequency so it is not too big (else no information by timeout).
  Then take average of past few iterations.
- Remove unused input args to run.py, or make a config file
- Handle case where if LR is on boundary, then we also need to run each momentum
  at that one. E.g. if best result is LR 0.01 and momentum 0.9, it's possible
  that an even better result would be LR 0.1 and momentum 0.6, but we would
  never test that in this case. So if it is on boundary, we should run a 2nd
  phase which runs all momentum values at that new LR
- Do not choose batch size but adjust batch size in proto slightly to properly
  divide # machines in a group
- Choose batch size  
- Make dynamic (re-evaluate every few iter)

"""

# ==============================================================================
# Parse arguments
# ==============================================================================

# ------------------------------------------------------------------------------
# Check for right number of args
# ------------------------------------------------------------------------------
expected_num_args = 2
if len(sys.argv) not in [expected_num_args, expected_num_args+1]:
    print '''
Usage:  >>> python  opt_main.py  config.txt

Example config.txt:

  solver_template = files/solver.prototxt      
  group_size_list = 32,128                          
  time_per_exp_short = 120                    
  time_per_exp_long  = 120                    


'''
    sys.exit(0)
# Pass in a D or d at the end to run in debug
DEBUG = False
if len(sys.argv) == expected_num_args+1 and sys.argv[expected_num_args] in ['d','D']:
    DEBUG = True

# ------------------------------------------------------------------------------
# Read params
# ------------------------------------------------------------------------------

# Params we need to read
solver_template = ''
group_size_list = []
time_per_exp_short = 0
time_per_exp_long = 0

# Read the experiment parameters
def parse_val(line, type_to_convert_to=str):
    return type_to_convert_to(line.split('=')[-1].strip())
def parse_list(line, type_to_convert_to=str):
    parsed_list_str = line.split('=')[-1].strip()
    parsed_list_str_type = []
    if ',' in parsed_list_str:
        for s in parsed_list_str.split(','):
            parsed_list_str_type.append(type_to_convert_to(s.strip()))
    else:
        parsed_list_str_type.append(type_to_convert_to(parsed_list_str))
    assert parsed_list_str_type
    return parsed_list_str_type
config_file = open(sys.argv[1])
for line in config_file:
    if 'solver_template' in line:
        solver_template = parse_val(line, str)
    elif 'group_size_list' in line:
        group_size_list = parse_list(line, int)
    elif 'time_per_exp_short' in line:
        time_per_exp_short = parse_val(line, int)
    elif 'time_per_exp_long' in line:
        time_per_exp_long = parse_val(line, int)
config_file.close()

# Make sure we got params we need
assert solver_template
# assert len(group_size_list) == 1     # For now optimize 1 group_size at a time
assert time_per_exp_short
# assert time_per_exp_long
solver_name = solver_template.split('/')[-1].split('.')[0]
SOLVER_DIR = 'solvers_' + solver_name
os.system('mkdir ' + SOLVER_DIR)


# ------------------------------------------------------------------------------
# Optimize over these:
# ------------------------------------------------------------------------------
momentum_list = []
LR_list = []

# Parse some parameters:
# - initial LR, used as a guess for tuning (note: not assuming this is tuned for 1 machine)
# - increment: used to parse log file at the end
initial_LR = 0
increment = 0
f = open(solver_template)
def parse_proto_val(line, type_to_convert_to):
    return type_to_convert_to( line.strip().split(':')[-1].strip() )
for line in f:
    if 'base_lr' in line:
        initial_LR = parse_proto_val(line, float)
    elif 'display' in line:
        increment = parse_proto_val(line, int)
f.close()
assert initial_LR and initial_LR > 0
assert increment


# ------------------------------------------------------------------------------
# For now these ones will be  hard-coded
# ------------------------------------------------------------------------------
base_dir = '/home/software/dcct/experiments/'
hw_types = ['CPU']
if FCC:
    fc_type = 'many_fc'
    map_fcc_to_cc = '1'
else:
    fc_type = 'single_fc'   # 'single_fc' or 'many_fc'
    map_fcc_to_cc = '0'
script_name_suffix = ''
script_name = 'run' + script_name_suffix + '.py'
# max_parallel_runs = 52

# Future options:
#   - delay 1s between cc launch
#   - 1 core or all cores?
#   + run with LOG(INFO)
#   + name server_... dir based on run name


# ==============================================================================
# Helper functions
# ==============================================================================

# Wrapper so I can comment out for debugging
def run_cmd(cmd):
    if DEBUG:
        return
    else:
        os.system(cmd)


# Read in the solver template and fill it in
# There will be these lines:
#   base_lr: ''' + str(LR) + '''
#   momentum: ''' + str(momentum) + '''
#   random_seed: ''' + str(random_seed) + '''
def make_solver(momentum, LR, fname, random_seed):
    
    # Read in the file and swap in the parameters
    output_str = ''
    f = open(solver_template)
    found_LR = False
    found_m  = False
    for line in f:
        if 'base_lr' in line and line.strip()[0] != '#':
            assert not found_LR
            output_str += ( 'base_lr: ' + str(LR) + "\n" )
            found_LR = True
        elif 'momentum' in line and line.strip()[0] != '#':
            assert not found_m
            output_str += ( 'momentum: ' + str(momentum) + "\n" )
            found_m  = True
        elif 'random_seed' in line and line.strip()[0] != '#':
            # We will insert our own later
            pass
        else:
            output_str += line
    assert found_LR
    assert found_m
    output_str += ( 'random_seed: ' + str(random_seed) + "\n" )
    f.close()
    
    # Write to a new file
    f = open(fname, 'w')
    f.write(output_str)
    f.close()


# Launch a run
# Currently serial but maybe we will want to do in parallel later
def run(group_size, hw_type, experiment_label, First, momentum, LR, seed, output_dir_base, run_time):

    # Check if we should make a new lmdb
    if First and MAKE_LMDB_FIRST:
        skip_string = ''
    else:
        skip_string = 's'

    run_id = '_'+ experiment_label + '.m' + str(momentum) + '.LR' + str(LR)
    fname = SOLVER_DIR + '/solver' + run_id + '.prototxt'
    
    # Make solver
    make_solver(momentum, LR, fname, seed)
    
    # Create the command to run
    logfile_out = 'logs/log.' + str(group_size) + 'mpg.' + hw_type + run_id
    run_experiment_command = 'python ' + script_name + ' ' + fname + ' example/machine_list.txt ' + str(group_size) + ' ' + hw_type + ' ' + fc_type + ' ' + map_fcc_to_cc + ' ' + output_dir_base + ' ' + skip_string + ' > ' + logfile_out + ' 2>&1'

    # Extra commands to wait and then kill servers
    if First:
        run_time += EXTRA_TIME_FIRST
    extra_cmds = ['sleep ' + str(run_time),
    'echo \'    Ending current run\'',
    'bash kill_servers' + script_name_suffix + '.sh',
    'sleep 10',
    'bash kill_servers' + script_name_suffix + '.sh']
        
    # Run commands
    print '[' + str(run_time/60) + ' min] ' + run_experiment_command
    run_cmd(run_experiment_command)
    # Wait for the command to finish and then kill the servers
    for extra_cmd in extra_cmds:
        run_cmd(extra_cmd)

    # Return the directory, which we can get from parsing the output file
    #   Writing input prototxt files to /path/server_input_files-2016-01-22-21-34-22/
    output_dir = ''
    f = open(logfile_out)
    for line in f:
        if 'Writing input prototxt files to' in line:
            output_dir = line.strip().split()[-1]
            break
    f.close()
    assert output_dir
    return output_dir
    

# Launch a run
# Currently serial but maybe we will want to do in parallel later
def print_run_cmd(group_size, hw_type, experiment_label, First, momentum, LR, seed, output_dir_base, run_time):

    skip_string = 's'
    run_id = '_'+ experiment_label + '.m' + str(momentum) + '.LR' + str(LR)
    fname = SOLVER_DIR + '/solver' + run_id + '.prototxt'
    logfile_out = 'logs/log.' + str(group_size) + 'mpg.' + hw_type + run_id
    run_experiment_command = 'python ' + script_name + ' ' + fname + ' example/machine_list.txt ' + str(group_size) + ' ' + hw_type + ' ' + fc_type + ' ' + map_fcc_to_cc + ' ' + output_dir_base + ' ' + skip_string + ' > ' + logfile_out + ' 2>&1'
    print '  ' + run_experiment_command
    

# ==============================================================================
# Main script
# ==============================================================================

def print_estimation_time(hw_types, random_seeds, group_size, momentum_list, LR_list, time_per_exp):
    time_for_1_run = int( time_per_exp + 15 + 15 )
    time_estimate = time_for_1_run*len(hw_types)*len(hw_types)*len(random_seeds)*len(momentum_list)*len(LR_list)
    time_estimate /= 60 # minutes
    if time_estimate > 60:
        print 'Estimated runtime: ' + str(time_estimate/60) + ' hours and ' + str(time_estimate%60) + ' minutes'
    else:
        print 'Estimated runtime: ' + str(time_estimate) + ' minutes'

# Only generate LMDB once for each group_size
First_time_for_this_group_size = {}
for group_size in group_size_list:
    First_time_for_this_group_size[group_size] = True

# Iterate over each group_size setting and optimize each separately
for group_size in group_size_list:

    print ''
    print '-----------------------------------------------------------------------------------' 
    print 'Beginning optimization for ' + str(group_size) + ' machines per group' 
    print '-----------------------------------------------------------------------------------' 

    EXP_NAME = solver_name + '_' + str(group_size) + 'mpg'  # Parse the name of the solver for log file names

    # The optimization procedure consists of a number of iteration phases
    # Each phase we will zoom in on the optimal parameters
    phase = 0
    momentum_list = momentum_list_phase_0
    # LR_list = [initial_LR*10., initial_LR, initial_LR/10.]
    LR_list = [0.01, 0.001, 0.0001, 0.00001]

    final_tuned_best_s  = None
    final_tuned_best_m  = None
    final_tuned_best_LR = None

    while True:
        # Estimate runtime for this phase
        print "\n" + '**********************************************************' + "\n" + 'Beginning tuning phase ' + str(phase) + "\n"
        if phase == 0:
            time_per_exp = time_per_exp_short
        else:
            assert phase == 1
            time_per_exp = time_per_exp_long
            
        if time_per_exp == 0:
            print '  Skipping phase, time set to 0'
            break
        
        print_estimation_time(hw_types, random_seeds, group_size, momentum_list, LR_list, time_per_exp)
        print '  Momentum: ' + str(momentum_list)
        print '  LR:       ' + str(LR_list)
        print '  seeds:    ' + str(random_seeds)

        # Keep a map from dirname to m and LR
        # map_dir_to_m  = {}
        # map_dir_to_LR = {}
        
        # Also in case there was an interruption or we just want to start the optimization
        # from a checkpoint, we can check if this has been run already
        list_of_output_dir = []
        experiment_dir = base_dir + '/' + EXP_NAME + '_PHASE_' + str(phase) + '/'
        num_experiments_to_skip = 0
        # Check if this exists
        if os.path.isdir(experiment_dir):
            # Count the number of experiments in here already
            num_experiments_to_skip = len(os.listdir(experiment_dir))
            for subdir in os.listdir(experiment_dir):
                list_of_output_dir.append(experiment_dir + subdir + '/')
            
        # Now run actual commands
        experiment_count = 0
        for hw_type in hw_types:
            for s in random_seeds:
                print "\n" + 'Running seed ' + str(s)
                for momentum in momentum_list:
                    for LR in LR_list:
                        if experiment_count >= num_experiments_to_skip:
                            output_dir = run(group_size, hw_type, EXP_NAME + '.PHASE' + str(phase) + '.seed' + str(s), First_time_for_this_group_size[group_size], momentum, LR, s, experiment_dir, time_per_exp)
                            First_time_for_this_group_size[group_size] = False
                            list_of_output_dir.append(output_dir)
                            # map_dir_to_m[output_dir]  = momentum
                            # map_dir_to_LR[output_dir] = LR
                        else:
                            print '  Found m=' + str(momentum) + ' LR=' + str(LR) + ', skipping command:'
                            print_run_cmd(group_size, hw_type, EXP_NAME + '.PHASE' + str(phase) + '.seed' + str(s), First_time_for_this_group_size[group_size], momentum, LR, s, experiment_dir, time_per_exp)
                            
                        experiment_count += 1
    
        print "\nDone running seeds\n"
        
        # Repeat for a number of final losses, but stop at the first one (only care about convergence)
        # SHADJIS TODO: Note here I am running them all, storing the dirs in a list, and then parsing them all
        # Another way could be to just parse each one as it finishes -- then we know the momentum, LR, etc.
        # I.e. these 2 loops should probably be merged
        final_losses = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0]
        for final_loss in final_losses:

            print 'Checking for final loss = ' + str(final_loss) + '...'
            
            # Store the results into a 2D array
            m_and_LR_to_iter = {}
            for m in momentum_list:
                m_and_LR_to_iter[m] = {}
                
            # Print some output as well
            output_lines = []
            best_iter = 100000000000
            best_str = ''
            best_s = None
            best_m = None
            best_LR = None

            # Read each output log and parse the #iter to this loss
            for experiment_dir in list_of_output_dir:
                if FCC:
                    f = open(experiment_dir + '/fc_compute_server.0.cfg.out')   # SHADJIS TODO: Should open each and average (else favors fewer fcc)
                else:
                    f = open(experiment_dir + '/fc_server.cfg.out')
                lines = f.read().strip().split("\n")
                    
                # Do lots of assertions (this can be removed later)
                # Read the output log and ensure it matches from above
                # E.g.:
                #   base_lr: 0.1
                #   momentum: 0.6
                #   weight_decay: 5e-05
                base_lr = ''
                momentum = ''
                weight_decay = ''
                random_seed = ''
                for line in lines:
                    if 'base_lr' in line:
                        base_lr = line.strip().split()[1]
                    if 'momentum' in line:
                        momentum = line.strip().split()[1]
                    if 'weight_decay' in line:
                        weight_decay = line.strip().split()[1]
                    if 'random_seed' in line:
                        random_seed = line.strip().split()[1]
                    if 'GROUPSIZE' in line:
                        assert group_size == int(line.strip().split()[-1])
                if not base_lr or not momentum or not weight_decay or not random_seed:
                    print 'ERROR in ' + experiment_dir + '/fc_server.cfg.out'
                # assert map_dir_to_m[experiment_dir]  == float(momentum)
                # assert map_dir_to_LR[experiment_dir] == float(LR)
                
                # Sometimes this did not run, check if the last line is actual data
                if 'SOFTMAX' in lines[-1] or 'my_create_zmq' in lines[-1]:
                    print "\t".join([random_seed, momentum, base_lr, weight_decay, 'ERROR ' + experiment_dir])
                    continue
                    
                # For each line, parse the loss
                # Print the first iter where loss < final_loss
                iter_below_final_loss = None
                loss_below_final_loss = None
                started_iterations = False
                # If this line has a loss, append it to a list so we can calculate running average
                running_avg_list = []
                final_iter = ''
                for line in lines:
                    if line.strip().split():
                        if not started_iterations:
                            if line.strip().split()[0] == str(increment):
                                started_iterations = True
                            else:
                                continue
                        loss_this_iter = float(line.strip().split()[2])
                        final_iter = line.strip().split()[0]
                        running_avg_list.append(loss_this_iter)
                        if len(running_avg_list) > running_avg:
                            assert len(running_avg_list) == running_avg + 1
                            running_avg_list = running_avg_list[1:]
                        if len(running_avg_list) == running_avg and (sum(running_avg_list) /len(running_avg_list)) < final_loss:
                            iter_below_final_loss = int(line.strip().split()[0])
                            loss_below_final_loss = (sum(running_avg_list) /len(running_avg_list))
                            break
                # Print this row
                if not iter_below_final_loss:
                    row = "\t".join([random_seed, momentum, base_lr, weight_decay, "DID NOT REACH (best was " + str((sum(running_avg_list) /len(running_avg_list))) + ' at ' + str(final_iter) + ")"])
                else:
                    row = "\t".join([random_seed, momentum, base_lr, weight_decay, str(iter_below_final_loss), str(loss_below_final_loss)])
                    m_and_LR_to_iter[float(momentum)][float(base_lr)] = iter_below_final_loss
                    if iter_below_final_loss < best_iter:
                        best_iter = iter_below_final_loss
                        best_str = row + "\t" + experiment_dir
                        best_s  = int(random_seed)
                        best_m  = float(momentum)
                        best_LR = float(base_lr)
                output_lines.append(row)
                f.close()
            # If anything reached this loss, then break out of final loss loop
            if best_str:
                print "\t".join(['seed', 'momentum', 'LR', 'Iter', 'end loss'])
                print "\n".join(list(sorted(output_lines)))
                print 'Best:'
                print best_str
        
                # Now we have m_and_LR_to_iter for each m / LR and also the best
                # SHADJIS TODO: m_and_LR_to_iter currently unused, but can use it later for more advanced tuning
                # SHADJIS TODO: Also use fewer seeds (e.g. only best from above)
                
                if phase == 0:
                    # Pick new momentum list:
                    if best_m == 0.0:
                        momentum_list = [0.0, 0.1, 0.2]
                    elif best_m == 0.3:
                        momentum_list = [0.1, 0.2, 0.3, 0.4, 0.5]
                    elif best_m == 0.6:
                        momentum_list = [0.4, 0.5, 0.6, 0.7, 0.8]
                    else:
                        assert best_m == 0.9
                        momentum_list = [0.7, 0.8, 0.9]
                        
                    # Pick a new LR list
                    #if best_LR == LR_list[0]:
                    #    LR_list = [LR_list[0]*10., LR_list[0]]
                    #elif best_LR == LR_list[-1]:
                    #    LR_list = [LR_list[-1], LR_list[-1]/10.]
                    #else:
                    #    assert best_LR == LR_list[1]
                    #    LR_list = [LR_list[1]]
                    LR_list = [best_LR]

                else:
                    assert phase == 1
                    final_tuned_best_s  = best_s
                    final_tuned_best_m  = best_m
                    final_tuned_best_LR = best_LR
                    
                # Break out of loop
                break
        
        if phase == 1:
            break
        phase += 1

    print "\n" + 'Experiment complete, final tuned result for ' + str(group_size) + ' machines per group:'
    # print '  s*  = ' + str(final_tuned_best_s)
    print '  m*  = ' + str(final_tuned_best_m)
    print '  LR* = ' + str(final_tuned_best_LR)
    
    # Now run a final experiment with these
    if final_time > 0:
        print 'Running for ' + str(final_time) + ' seconds...'
        experiment_dir = base_dir + '/' + EXP_NAME + '_FINAL_PHASE/'
        for hw_type in hw_types:
            output_dir = run(group_size, hw_type, EXP_NAME + '.FINAL_PHASE.seed' + str(final_tuned_best_s), First_time_for_this_group_size[group_size], final_tuned_best_m, final_tuned_best_LR, final_tuned_best_s, experiment_dir, final_time)
            print 'See ' + output_dir
