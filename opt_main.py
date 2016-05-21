#!/usr/bin/python

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
hw_type = '4GPU' #'4GPU' 'GPU'

random_seeds_phase_0 = [1]  # Seeds matter a lot, but will not search now (save time)
EXTRA_TIME_FIRST = 0
MAKE_LMDB_FIRST = False
FCC = False

# leave these empty to let the system guess
momentum_list_phase_0 = []
LR_list_phase_0 = []


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


def read_output_by_lines(run_dir):
    if FCC:
        fname = run_dir + '/fc_compute_server.0.cfg.out'   # SHADJIS TODO: Should open each and average (else favors fewer fcc)
    else:
        fname = run_dir + '/fc_server.cfg.out'
    f = open(fname)
    lines = f.read().strip().split("\n")
    f.close()
    return lines


def get_list_of_all_losses(lines, increment):
    # For each line, parse the loss
    started_iterations = False
    # If this line has a loss, append it to a list
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
    return list_of_all_losses


# Wrapper so I can comment out for debugging
def run_cmd(cmd):
    if DEBUG:
        return
    else:
        os.system(cmd)


def contains_nan(L):
    import math
    for l in L:
        if math.isnan(l):
            return True
    return False


# Wrapper so I can comment out for debugging
def get_lr_m_s(lines, group_size):
    # Do lots of assertions (this can be removed later)
    # Read the output log and ensure it matches from above
    # E.g.:
    #   base_lr: 0.1
    #   momentum: 0.6
    base_lr = ''
    momentum = ''
    random_seed = ''
    for line in lines:
        if 'base_lr' in line:
            base_lr = line.strip().split()[1]
        if 'momentum' in line:
            momentum = line.strip().split()[1]
        if 'random_seed' in line:
            random_seed = line.strip().split()[1]
        if 'GROUPSIZE' in line:
            assert group_size == int(line.strip().split()[-1])
    return base_lr, momentum, random_seed


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

def print_estimation_time(random_seeds, group_size, momentum_list, LR_list, time_per_exp):
    time_for_1_run = int( time_per_exp + 15 + 15 )
    time_estimate = time_for_1_run*len(random_seeds)*len(momentum_list)*len(LR_list)
    time_estimate /= 60 # minutes
    if time_estimate > 60:
        print 'Estimated runtime: ' + str(time_estimate/60) + ' hours and ' + str(time_estimate%60) + ' minutes'
    else:
        print 'Estimated runtime: ' + str(time_estimate) + ' minutes'

# Only generate LMDB once for this cluster
First_Run = True

best_group_size = None
best_group_size_s = None
best_group_size_m = None
best_group_size_LR = None
best_loss_across_staleness = 10000000000

# Iterate over each group_size setting and optimize each separately
# Iterate in order from low staleness to high staleness (i.e. large to small groups)
# This is because we know that the optimal parameters will be smaller as S increases
best_LR_last_iteration = None
best_m_last_iteration = None
# For the seed, just run multiple seeds with no staleness and then use the best
# one for other staleness values
best_seed_last_iteration = None

# SHADJIS TODO: For now I assume the first iteration is a single group, i.e. I assume
# that when sorting group sizes from largest to smallest the largest group is all machines.
# This might not be true so can verify. But when running 1 group, don't tune m 
# Note: it is possible that for S=0, i.e. 1 group, the best momentum is not 0.9, e.g. it is 
# possible that LR 0.001, m 0.9 does not diverge, and also that LR 0.01, m 0.3 does not diverge.
# However our contribution is tuning parameters to compensate for staleness, so the optimizer
# can ignore tuning momentum for the case of no staleness to save time.
single_group = True

for group_size in reversed(sorted(group_size_list)):

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
        if single_group:
            # Optimization:
            # See comment above, if S=0 we can choose to skip momentum tuning
            momentum_list = [0.9]
        else:
            momentum_list = [0.0, 0.3, 0.6, 0.9]
        
    if LR_list_phase_0:
        LR_list = LR_list_phase_0
    else:
        if single_group:
            LR_list = [initial_LR*100., initial_LR*10., initial_LR, initial_LR/10., initial_LR/100.]
        else:
            LR_list = [best_LR_last_iteration, best_LR_last_iteration/10.]
    
    # Optimization:
    # For the first iteration, run multiple seeds
    # Then pick the best one for later runs
    if single_group:
        random_seeds = random_seeds_phase_0
    else:
        random_seeds = [best_seed_last_iteration]
        
    
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
        
        print_estimation_time(random_seeds, group_size, momentum_list, LR_list, time_per_exp)
        print '  Momentum: ' + str(momentum_list)
        print '  LR:       ' + str(LR_list)
        print '  seeds:    ' + str(random_seeds)

        # Check if any of these have been run already
        # This is useful in case there is an interruption and the experiment did not complete fully
        #
        # Iterate over the existing directories and make a list of the ones already run
        m_LR_s_to_output_dir = {}   # This one has keys which are strings
        # Check if any runs exist
        experiment_dir = base_dir + '/' + EXP_NAME + '_PHASE_' + str(phase) + '/'
        if os.path.isdir(experiment_dir):
            # Check if any of these have already run
            for subdir in os.listdir(experiment_dir):
                # Read the output file and check for errors
                full_experiment_dir = experiment_dir + subdir + '/'
                lines = read_output_by_lines(full_experiment_dir)
                base_lr, momentum, random_seed = get_lr_m_s(lines, group_size)
                list_of_all_losses = get_list_of_all_losses(lines, increment)
                # Check for errors
                if not base_lr or not momentum or not random_seed or not list_of_all_losses:
                    # This one didn't finish properly, so rerun it later
                    continue
                # Otherwise this one ran, so no need to run again
                m_LR_s_to_output_dir[(momentum, base_lr, random_seed)] = full_experiment_dir

        # Now we have a map (set) of the parameters which finished already
        # Run the remaining parameters
        # We will map each run to a loss
        m_LR_s_to_loss = {}   # This one has keys which are float float int (should make consistent with above map which is strings)
            
        # Print some output as well
        output_lines = []
        best_str = ''
        best_s = None
        best_m = None
        best_LR = None
        best_loss = 10000000000
        
        # Now run commands
        for s in random_seeds:
            print "\n" + 'Running seed ' + str(s)
        
            # Optimization: if we hit NaN for some parametes, no need to search larger parameters
            NaN_LR = None
            NaN_m = None

            # SHADJIS TODO: Optimization:
            # Iterate from high to low for LR and m and if it ever gets worse
            # (or e.g. only better by a small margin like 5%), then stop.
            # This is because we expect the best parameters to be larger than
            # the smallest parameters we check, so we can save time
            for LR in sorted(LR_list):  # Low to high
                for m in sorted(momentum_list): # Low to high
                
                    # Optimization:
                    # Skip this iteration if it will be Nan
                    # The LR check is redundant since it is in the outer loop, i.e. every LR will be >= NaN_LR
                    if NaN_LR is not None and NaN_m is not None:
                        if LR >= NaN_LR and m >= NaN_m:
                            print '  Skipping LR = ' + str(LR) + ', m = ' + str(m) + ', s = ' + str(s) + ' to avoid NaN'
                            continue
                
                    # Also skip this iteration if LR and m are larger than the previous iteration's optimal LR and m
                    # This is because we run from low to high staleness
                    if best_LR_last_iteration is not None and best_m_last_iteration is not None:
                        if LR > best_LR_last_iteration or (LR == best_LR_last_iteration and m > best_m_last_iteration):
                            print '  Skipping LR = ' + str(LR) + ', m = ' + str(m) + ', s = ' + str(s) + ' because of previous iteration (staleness)'
                            continue
                
                    # if this seed/momentum/LR ran already, skip it
                    if (str(m), str(LR), str(s)) in m_LR_s_to_output_dir.keys():
                        print_only = True
                        print '  Found m=' + str(m) + ' LR=' + str(LR) + ' s=' + str(s) + ', skipping command:'
                        run(group_size, hw_type, EXP_NAME + '.PHASE' + str(phase) + '.seed' + str(s), First_Run, m, LR, s, experiment_dir, time_per_exp, print_only)
                        full_experiment_dir = m_LR_s_to_output_dir[(str(m), str(LR), str(s))]
                    # otherwise run this command
                    else:
                        full_experiment_dir = run(group_size, hw_type, EXP_NAME + '.PHASE' + str(phase) + '.seed' + str(s), First_Run, m, LR, s, experiment_dir, time_per_exp)
                        First_Run = False

                    # Now the command has run, read the output log and parse the final (or average, etc.) loss
                    lines = read_output_by_lines(full_experiment_dir)
                    base_lr, momentum, random_seed = get_lr_m_s(lines, group_size)
                    list_of_all_losses = get_list_of_all_losses(lines, increment)
                    # Check for errors
                    if not base_lr or not momentum or not random_seed or not list_of_all_losses:
                        print "\t".join([random_seed, momentum, base_lr, 'ERROR ' + full_experiment_dir])
                        # We could break here because every larger momentum would be skipped for NaN,
                        # but by continuing instead it will print that it is skipping them
                        continue
                    assert base_lr == str(LR)
                    assert momentum == str(m)
                    assert random_seed == str(s)
                    
                    # Check also if there was a nan in this result
                    # If so, we know not to run a higher momentum, although 
                    if contains_nan(list_of_all_losses):
                        NaN_LR = LR
                        NaN_m = m
                        print '  NaN found for LR = ' + str(LR) + ', m = ' + str(m) + ', s = ' + str(s)
                        continue
                    
                    # Calculate average loss for run
                    average_loss = sum(list_of_all_losses) / float(len(list_of_all_losses))
                    row = "\t".join([random_seed, momentum, base_lr, str(average_loss)])
                    m_LR_s_to_loss[(float(momentum), float(base_lr), int(random_seed))] = average_loss
                    if average_loss < best_loss:
                        best_loss = average_loss
                        best_str = row + "\t" + full_experiment_dir
                        best_s  = int(random_seed)
                        best_m  = float(momentum)
                        best_LR = float(base_lr)
                    output_lines.append(row)
        
        # Every command has now run
        assert best_str
        print ''
        print "\t".join(['seed', 'momentum', 'LR', 'loss'])
        print "\n".join(list(sorted(output_lines)))
        print 'Best:'
        print best_str

        # Now we have m_LR_s_to_loss for each m / LR / s and also the best,
        # Just using the best_* parameters works well, but since we have
        # m_LR_s_to_loss we can pick parameters which are higher (e.g.
        # higher LR, higher m) ass long as the final loss is not much
        # worse (e.g. within 10%). This works better in the long-run.
        #
        # First iterate over LR and pick highest LR possible
        original_best_LR = best_LR
        original_best_m  = best_m
        for s in random_seeds:
            for LR in sorted(LR_list):  # Lowest to highest
                # only consider LR bigger than or equal to the best one
                if LR < original_best_LR:
                    continue
                for m in sorted(momentum_list):     # Lowest to highest
                    # at the same LR, only pick a larger m
                    if LR == original_best_LR and m <= original_best_m:
                        continue
                    # Check if it is within 10%
                    if (m,LR,s) in m_LR_s_to_loss.keys() and m_LR_s_to_loss[(m,LR,s)] < best_loss*1.1:
                        print 'Adjusting best to m = ' + str(m) + ', LR = ' + str(LR) + ', s = ' + str(s) + ', loss = ' + str(m_LR_s_to_loss[(m,LR,s)])
                        best_LR = LR
                        best_m  = m
                        best_s  = s
        
        if phase == 0:
            # Pick new momentum list:
            if best_m == 0.0:
                momentum_list = [0.0, 0.1, 0.2]
            elif best_m == 0.3:
                momentum_list = [0.1, 0.2, 0.3, 0.4, 0.5]
            elif best_m == 0.6:
                momentum_list = [0.4, 0.5, 0.6, 0.7, 0.8]
            else:
                # assert best_m == 0.9
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
            random_seeds = [best_s] # If we used more than 1 seed for previous phase, only need 1 for next phase
            
        # If running more than 2 phases, can made similar parameter adjustments for next phase here
        #elif...
       
    print "\n" + '**********************************************************'
    print 'Experiment complete, final tuned result for ' + str(group_size) + ' machines per group:'
    print '  s*  = ' + str(best_s)
    print '  m*  = ' + str(best_m)
    print '  LR* = ' + str(best_LR)
    
    # Now run a final experiment with these
    if time_per_exp_phase3 > 0:
        print 'Running for ' + str(time_per_exp_phase3) + ' seconds...'
        full_experiment_dir = base_dir + '/' + EXP_NAME + '_FINAL_PHASE/'
        # Check if this exists
        if os.path.isdir(full_experiment_dir):
            print '  Skipping, already ran'
            list_of_subdir = os.listdir(full_experiment_dir)
            if len(list_of_subdir) != 1:
                print 'Error -- please only put 1 directory in ' + full_experiment_dir + ' (to simplify parsing)'
                sys.exit(0)
            output_dir = full_experiment_dir + list_of_subdir[0]
        else:
            # Run the experiment
            list_of_subdir = run(group_size, hw_type, EXP_NAME + '.FINAL_PHASE.seed' + str(best_s), First_Run, best_m, best_LR, best_s, full_experiment_dir, time_per_exp_phase3)
        # Parse the output
        lines = read_output_by_lines(output_dir)
        if 'SOFTMAX' in lines[-1] or 'my_create_zmq' in lines[-1]:
            print '  Run failed, need to rerun!'
            continue
        list_of_all_losses = get_list_of_all_losses(lines, increment)
        # Calculate the average loss of the last few iterations, e.g. the last 5 or 10
        # (but make sure it is consistent across S, otherwise not a fair comparison)
        last_few_iter = 5
        assert last_few_iter > 0
        average_loss = sum(list_of_all_losses[-last_few_iter:]) / float(last_few_iter)
        print "\n" + 'Final loss for group size ' + str(group_size) + ' = ' + str(average_loss) + "\n"
    else:
        print 'Not running the best for longer, re-using the best from phase 1/2'
        average_loss = m_LR_s_to_loss[(best_m, best_LR, best_s)]
        print "\n" + 'Final loss for group size ' + str(group_size) + ' = ' + str(average_loss) + "\n"
    
    if average_loss < best_loss_across_staleness:            
        best_group_size = group_size
        best_loss_across_staleness = average_loss
        best_group_size_s = best_s
        best_group_size_m = best_m
        best_group_size_LR = best_LR
    
    # Done this group. If it was the first iteration, now it is not the first iteration anymore
    single_group = False
    best_LR_last_iteration = best_LR
    best_m_last_iteration = best_m
    best_seed_last_iteration = best_s

print ''
print 'Finished optimizer, best result is group size ' + str(best_group_size)
print 'Running this one *without a timeout* (Ctrl-C will (1) stop the job and (2) run kill script)'
full_experiment_dir = base_dir + '/' + EXP_NAME + '_OPTIMIZER_DECISION/'
if os.path.isdir(full_experiment_dir):
    print '  Skipping, already ran'
else:
    # Run the experiment
    output_dir = run(best_group_size, hw_type, EXP_NAME + '.OPTIMIZER_DECISION.seed' + str(best_group_size_s), First_Run, best_group_size_m, best_group_size_LR, best_group_size_s, full_experiment_dir, 600000)
