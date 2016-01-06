import os

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

hw_types = ['4GPU']
num_machines_per_group_list = [1,2,4,8]
num_group_list = [32,16,8,4]
num_runs = 2
time_per_exp = 3600*1
NUM_MACHINES = 32

#CPU
# hw_types = ['CPU']
# num_machines_per_group_list = [2, 16]
# num_group_list = [2, 16]
# num_runs = 2
# time_per_exp = 3600*8
# NUM_MACHINES = 32

#Staleness
# hw_types = ['4GPU']
# num_machines_per_group_list = [2]
# num_group_list = [16]
# num_runs = 1
# time_per_exp = 600
# NUM_MACHINES = 32


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

# Wrapper so I can comment out for debugging
def run_cmd(cmd):
    os.system(cmd)
    # return

# Run 30 min plus extra if fewer machines
#def get_run_time(num_convcompute):
#    return 3600 # Terminate all at 30 min, and sometimes slow to start

# Unused for now (but more runs should be needed for more groups?)
# We can do analysis of variance later and see
#def get_num_runs(num_groups, num_machine_per_grp, num_convcompute):
#    num_runs = 1
#    if num_groups <= 2:
#        if num_convcompute == 8:
#            num_runs = 2
#        elif num_convcompute > 8:
#            num_runs = 3
#        else:
#            num_runs = 1
#    elif num_groups <= 4:   # Run two runs (a and b)
#        num_runs = 2
#    else:                   # Run three runs (a,b,c)
#        num_runs = 3
#    return num_runs

def run(num_convcompute, num_machine_per_grp, hw_type, experiment_label):

    # Clear old lmdb
    # run_cmd('rm -rf ilsvrc12_train_lmdb_8_p*')
    # run_cmd('sleep 20')

    # Create the command to run. Will run multiple times if many machines (since variance)
    # num_groups = num_convcompute / num_machine_per_grp
    base_cmd = 'python run.py example/solver_template.prototxt example/m' + str(num_convcompute) + '.txt ' + str(num_machine_per_grp) + ' ' + hw_type + ' s > logs/log.' + str(num_convcompute) + '.' + str(num_machine_per_grp) + '.' + hw_type

    # Extra commands to wait and then kill servers
    run_time = time_per_exp#get_run_time(num_convcompute)
    extra_cmds = ['sleep ' + str(run_time),
    'bash kill_servers.sh',
    'sleep 10',
    'bash kill_servers.sh',
    'sleep 10']
        
    # Run commands
    cmd = base_cmd + '_'+ experiment_label + ' 2>&1'
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

for r in range(num_runs):
    for hw_type in hw_types:
        for num_machines_per_group in num_machines_per_group_list:
            for num_group in num_group_list:
                if num_group * num_machines_per_group != NUM_MACHINES:
                    continue
                num_conv_compute = num_group*num_machines_per_group
                time_for_1_run = time_per_exp#get_run_time(num_conv_compute) + 2     # 2 min to make lmdb and wait between launching
                time_for_1_run /= 60 # minutes
                est += time_for_1_run
if est > 60:
    print 'Estimated runtime: ' + str(est/60) + ' hours and ' + str(est%60) + ' minutes'
else:
    print 'Estimated runtime: ' + str(est) + ' minutes'

# Now run actual commands
for hw_type in hw_types:
    for r in range(num_runs):
        for num_machines_per_group in num_machines_per_group_list:
            print "\n" + 'Running ' + str(num_machines_per_group) + ' machine(s) per group'
            for num_group in num_group_list:
                if num_group * num_machines_per_group != NUM_MACHINES:
                    continue
                run(num_group * num_machines_per_group, num_machines_per_group, hw_type, 'FCCM' + str(r))

print "\n" + 'Experiment complete'
