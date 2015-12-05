import os

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

num_machines_per_group_list = [1,4,8]
num_group_list = [1,2,4,8,16]


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

# Wrapper so I can comment out for debugging
def run_cmd(cmd):
    os.system(cmd)
    # return

# Run 30 min plus extra if fewer machines
def get_run_time(num_convcompute):
    return 1800 + 1800/num_convcompute

def get_num_runs(num_groups, num_machine_per_grp, num_convcompute):
    num_runs = 1
    if num_groups <= 2:
        if num_convcompute == 8:
            num_runs = 2
        elif num_convcompute > 8:
            num_runs = 3
        else:
            num_runs = 1
    elif num_groups <= 4:   # Run two runs (a and b)
        num_runs = 2
    else:                   # Run three runs (a,b,c)
        num_runs = 3
    return num_runs

def run(num_convcompute, num_machine_per_grp):

    # Clear old lmdb
    run_cmd('rm -rf ilsvrc12_train_lmdb_8_p*')
    run_cmd('sleep 20')

    # Create the command to run. Will run multiple times if many machines (since variance)
    num_groups = num_convcompute / num_machine_per_grp
    base_cmd = 'python run.py example/solver_template.prototxt example/m' + str(num_convcompute) + '.txt ' + str(num_machine_per_grp) + ' > logs/log.' + str(num_convcompute) + '.' + str(num_machine_per_grp)

    num_runs = get_num_runs(num_groups, num_machine_per_grp, num_convcompute)
    if num_runs == 1:
        cmds = [base_cmd + ' 2>&1']
    elif num_runs == 2:
        cmds = [base_cmd + 'a 2>&1',
        base_cmd + 'b 2>&1']
    else:
        cmds = [base_cmd + 'a 2>&1',
        base_cmd + 'b 2>&1',
        base_cmd + 'c 2>&1']
    
    # Extra commands to wait and then kill servers
    run_time = get_run_time(num_convcompute)
    extra_cmds = ['sleep ' + str(run_time),
    'bash kill_servers.sh',
    'sleep 10',
    'bash kill_servers.sh',
    'sleep 10']
        
    # Run commands
    for cmd in cmds:
        print '[' + str(run_time/60) + ' min] ' + cmd
        run_cmd(cmd)
        # Wait for the command to finish and then kill the servers
        for extra_cmd in extra_cmds:
            print '    ' + extra_cmd
            run_cmd(extra_cmd)


# ------------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------------

# First estimate runtime
est = 0
for num_machines_per_group in num_machines_per_group_list:
    for num_group in num_group_list:
        num_conv_compute = num_group*num_machines_per_group
        time_for_1_run = get_run_time(num_conv_compute) + 2     # 2 min to make lmdb
        time_for_1_run /= 60 # minutes
        num_runs = get_num_runs(num_group, num_machines_per_group, num_conv_compute)
        est += time_for_1_run*num_runs
if est > 60:
    print 'Estimated runtime: ' + str(est/60) + ' hours and ' + str(est%60) + ' minutes'
else:
    print 'Estimated runtime: ' + str(est) + ' minutes'

# Now run actual commands
for num_machines_per_group in num_machines_per_group_list:

    print "\n" + 'Running ' + str(num_machines_per_group) + ' machine(s) per group'
    for num_group in num_group_list:
        run(num_group * num_machines_per_group, num_machines_per_group)

print "\n" + 'Experiment complete'
