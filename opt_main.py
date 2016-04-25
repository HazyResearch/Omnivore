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

# config file provides all parameters, but here are some extra ones

base_dir = '/home/software/dcct/experiments/'
hw_types = ['4GPU']

random_seeds_phase_0 = [1]  # Seeds matter a lot, but will not search now (save time)
EXTRA_TIME_FIRST = 0
MAKE_LMDB_FIRST = False
FCC = False

# leave these empty to let the system guess
momentum_list_phase_0 = []
LR_list_phase_0 = [0.001, 0.01]


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
  group_size_list = 1,2,4,8,16,32
  time_per_exp_phase1 = 60
  time_per_exp_phase2 = 0
  time_per_exp_phase3 = 300


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
time_per_exp_phase1 = 0
time_per_exp_phase2 = 0
time_per_exp_phase3 = 0

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
    elif 'time_per_exp_phase1' in line:
        time_per_exp_phase1 = parse_val(line, int)
    elif 'time_per_exp_phase2' in line:
        time_per_exp_phase2 = parse_val(line, int)
    elif 'time_per_exp_phase3' in line:
        time_per_exp_phase3 = parse_val(line, int)
config_file.close()

# Make sure we got params we need
assert solver_template
# assert len(group_size_list) == 1     # For now optimize 1 group_size at a time
assert time_per_exp_phase1
# assert time_per_exp_phase2
solver_name = solver_template.split('/')[-1].split('.')[0]
SOLVER_DIR = 'solvers_' + solver_name
os.system('mkdir -p ' + SOLVER_DIR)


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
def run(group_size, hw_type, experiment_label, First, momentum, LR, seed, output_dir_base, run_time, print_only = False):

    # Check if we should make a new lmdb
    if First and MAKE_LMDB_FIRST:
        skip_string = ''
    else:
        skip_string = 's'

    run_id = '_'+ experiment_label + '.m' + str(momentum) + '.LR' + str(LR)
    fname = SOLVER_DIR + '/solver' + run_id + '.prototxt'
    
    # Create the command to run
    os.system('mkdir -p logs')
    logfile_out = 'logs/log.' + hw_type + run_id
    run_experiment_command = 'python ' + script_name + ' ' + fname + ' example/machine_list.txt ' + str(group_size) + ' ' + hw_type + ' ' + fc_type + ' ' + map_fcc_to_cc + ' ' + output_dir_base + ' ' + skip_string + ' > ' + logfile_out + ' 2>&1'

    if print_only:
        print '  ' + run_experiment_command
        return None
    
    # Make solver
    make_solver(momentum, LR, fname, seed)
    
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
    

# ==============================================================================
# Main script
# ==============================================================================

def print_estimation_time(hw_types, random_seeds, group_size, momentum_list, LR_list, time_per_exp):
    time_for_1_run = int( time_per_exp + 15 + 15 )
    time_estimate = time_for_1_run*len(hw_types)*len(random_seeds)*len(momentum_list)*len(LR_list)
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
    if momentum_list_phase_0:
        momentum_list = momentum_list_phase_0
    else:
        momentum_list = [0.0, 0.3, 0.6, 0.9]
        
    if LR_list_phase_0:
        LR_list = LR_list_phase_0
    else:
        LR_list = [initial_LR*10., initial_LR, initial_LR/10.]
    
    random_seeds = random_seeds_phase_0
    
    for phase in [0,1]:
    
        # Estimate runtime for this phase
        print "\n" + '**********************************************************' + "\n" + 'Beginning tuning phase ' + str(phase) + "\n"
        if phase == 0:
            time_per_exp = time_per_exp_phase1
        else:
            assert phase == 1
            time_per_exp = time_per_exp_phase2
            
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
                            print_only = True
                            print '  Found m=' + str(momentum) + ' LR=' + str(LR) + ', skipping command:'
                            run(group_size, hw_type, EXP_NAME + '.PHASE' + str(phase) + '.seed' + str(s), First_time_for_this_group_size[group_size], momentum, LR, s, experiment_dir, time_per_exp, print_only)
                            
                        experiment_count += 1
    
        print "\nDone running seeds\n"
        
        # Now parse the logs above
        # Store the results into a 2D array,  A[m][LR] = final_loss
        m_and_LR_to_loss = {}
        for m in momentum_list:
            m_and_LR_to_loss[m] = {}
            
        # Print some output as well
        output_lines = []
        best_str = ''
        best_s = None
        best_m = None
        best_LR = None
        best_loss = 10000000000

        # Read each output log and parse the final (or average, etc.) loss
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
            started_iterations = False
            # If this line has a loss, append it to a list so we can calculate running average
            list_of_all_losses = []
            for line in lines:
                if line.strip().split():
                    if not started_iterations:
                        if line.strip().split()[0] == str(increment):
                            started_iterations = True
                        else:
                            continue
                    loss_this_iter = float(line.strip().split()[2])
                    list_of_all_losses.append(loss_this_iter)

            # Print this row
            if not list_of_all_losses:
                print 'MISSING: ' + experiment_dir
                continue
            average_loss = sum(list_of_all_losses) / float(len(list_of_all_losses))
            row = "\t".join([random_seed, momentum, base_lr, weight_decay, str(average_loss)])
            m_and_LR_to_loss[float(momentum)][float(base_lr)] = average_loss
            if average_loss < best_loss:
                best_loss = average_loss
                best_str = row + "\t" + experiment_dir
                best_s  = int(random_seed)
                best_m  = float(momentum)
                best_LR = float(base_lr)
            output_lines.append(row)
            f.close()
        
        assert best_str
        print ''
        print "\t".join(['seed', 'momentum', 'LR', 'loss'])
        print "\n".join(list(sorted(output_lines)))
        print 'Best:'
        print best_str

        # Now we have m_and_LR_to_loss for each m / LR and also the best
        # Just using the best_* parameters works well, but since we have
        # m_and_LR_to_loss we can pick parameters which are higher (e.g.
        # higher LR, higher m) ass long as the final loss is not much
        # worse (e.g. within 10%). This works better in the long-run.
        #
        # First iterate over LR and pick highest LR possible
        assert len(random_seeds) == 1 # Handle this case later
        original_best_LR = best_LR
        original_best_m  = best_m
        for LR in sorted(LR_list):  # Lowest to highest
            # only consider LR bigger than or equal to the best one
            if LR < original_best_LR:
                continue
            for m in sorted(momentum_list):     # Lowest to highest
                # at the same LR, only pick a larger m
                if LR == original_best_LR and m <= original_best_m:
                    continue
                # Check if it is within 10%
                if m_and_LR_to_loss[m][LR] < best_loss*1.1:
                    print 'Adjusting best to m = ' + str(m) + ', LR = ' + str(LR) + ', loss = ' + str(m_and_LR_to_loss[m][LR])
                    best_LR = LR
                    best_m  = m
        
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
            random_seeds = [best_s]
            
        # If running more than 2 phases, can made similar parameter adjustments for next phase here
        #elif...
        

    print "\n" + 'Experiment complete, final tuned result for ' + str(group_size) + ' machines per group:'
    print '  s*  = ' + str(best_s)
    print '  m*  = ' + str(best_m)
    print '  LR* = ' + str(best_LR)
    
    # Now run a final experiment with these
    if time_per_exp_phase3 > 0:
        print 'Running for ' + str(time_per_exp_phase3) + ' seconds...'
        experiment_dir = base_dir + '/' + EXP_NAME + '_FINAL_PHASE/'
        for hw_type in hw_types:
            output_dir = run(group_size, hw_type, EXP_NAME + '.FINAL_PHASE.seed' + str(best_s), First_time_for_this_group_size[group_size], best_m, best_LR, best_s, experiment_dir, time_per_exp_phase3)
            print 'See ' + output_dir
    else:
        print 'Not running the best for longer, re-using the best from phase 1 or 2'
