# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

NUM_COMPUTE = 1
dir = str(NUM_COMPUTE) + '_profile'
out_name = dir  # No ext needed, automatically will be .png


# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import sys, os
import datetime
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def fsec(s):
  hours, minutes, seconds = [float(val) for val in s.split(':')]
  return int((hours*3600+minutes*60+seconds)*1000)


# ------------------------------------------------------------------------------
# Read all files to get size of graph
# ------------------------------------------------------------------------------

min_times = []
max_times = [0]*(2+NUM_COMPUTE)
server = 0

# Read each log file to get the min time in each

# Parse conv model server
first = True
for l in open('../' + dir + '/conv_model_server.cfg.out'):
  l = l.rstrip()
  if first and ('~~~~ ENTER STATE' in l):
    min_times.append( fsec(l.split(' ')[1]) )
    first = False
  elif ('~~~~ EXIT STATE' in l):
    max_times[server] = fsec(l.split(' ')[1])
server += 1
# print '.'

# Parse fc server
first = True
for l in open('../' + dir + '/fc_server.cfg.out'):
  l = l.rstrip()
  if first and ('~~~~ ENTER STATE' in l):
    min_times.append( fsec(l.split(' ')[1]) )
    first = False
  elif ('~~~~ EXIT STATE' in l):
    max_times[server] = fsec(l.split(' ')[1])
server += 1
# print '.'

# Parse conv compute servers
for worker in range(NUM_COMPUTE):
  first = True
  for l in open('../' + dir + '/conv_compute_server.%d.cfg.out' % worker):
    l = l.rstrip()
    if first and ('~~~~ ENTER STATE' in l):
      min_times.append( fsec(l.split(' ')[1]) )
      first = False
    elif ('~~~~ EXIT STATE' in l):
      max_times[server] = fsec(l.split(' ')[1])
  server += 1
  # print '.'

assert len(min_times) == 2+NUM_COMPUTE
assert len(max_times) == 2+NUM_COMPUTE
MIN_TIME = min(min_times)
MAX_TIME = min(max_times)
END_TIME = MAX_TIME#MIN_TIME + 1000
# print MIN_TIME
# print MAX_TIME
# print END_TIME
  

# ------------------------------------------------------------------------------
# Set up graphics
# ------------------------------------------------------------------------------

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.set_ylim([MIN_TIME,END_TIME])
ax1.set_xlim([0,0.6+0.3*NUM_COMPUTE])
ax1.yaxis.grid(True)
# ax1.yaxis.set_ticks([i*10.0 for i in range(649810,789810)])

colors = {
  # FC
  'Read msg': "red",
  'Update input layer': "blue",
  'FC Get Grad': "blue",
  'FC FW': "pink",
  'FC BW': "pink",
  'ACC': "yellow",
  
  # Conv Model
  'Read corpus': "red",
  'Update Model': "blue",
  'Update gradients': "blue",
  'Copy FW': "yellow",
  'Copy BW': "yellow",
  'Conv FW': "pink",
  'Conv BW': "pink",
  'Read msg': "black",

  # Conv Compute
  'Copy Model': "red",
}


# ------------------------------------------------------------------------------
# Generate Plots
# ------------------------------------------------------------------------------

# Parse conv model server
lasttime = None
for l in open('../' + dir + '/conv_model_server.cfg.out'):
  l = l.rstrip()
  if ('~~~~ ENTER STATE' in l) and ('IDLE' not in l):
    t = fsec(l.split(' ')[1])
    if t >= END_TIME: break
    lasttime = t
  elif ('~~~~ EXIT STATE' in l) and ('IDLE' not in l) and (lasttime != None):
    t = fsec(l.split(' ')[1])
    if t >= END_TIME: break
    m = re.search('EXIT STATE (.*?)$', l)
    # print t-lasttime, m.group(1)
    ax1.add_patch(patches.Rectangle((0.1, lasttime), 0.2, t-lasttime, color=colors[m.group(1)]))

# Parse fc server
lasttime = None
for l in open('../' + dir + '/fc_server.cfg.out'):
  l = l.rstrip()
  if ('~~~~ ENTER STATE' in l) and ('IDLE' not in l):
    t = fsec(l.split(' ')[1])
    if t >= END_TIME: break
    lasttime = t
  elif ('~~~~ EXIT STATE' in l) and ('IDLE' not in l) and (lasttime != None):
    t = fsec(l.split(' ')[1])
    if t >= END_TIME: break
    m = re.search('EXIT STATE (.*?)$', l)
    print t-lasttime, m.group(1)
    ax1.add_patch(patches.Rectangle((0.3, lasttime), 0.2, t-lasttime, color=colors[m.group(1)]))

# Parse conv compute servers
for worker in range(NUM_COMPUTE):
  lasttime = None
  for l in open('../' + dir + '/conv_compute_server.%d.cfg.out' % worker):
    l = l.rstrip()
    if ('~~~~ ENTER STATE' in l) and ('IDLE' not in l):
      t = fsec(l.split(' ')[1])
      if t >= END_TIME: break
      lasttime = t
    elif ('~~~~ EXIT STATE' in l) and ('IDLE' not in l) and (lasttime != None):
      t = fsec(l.split(' ')[1])
      if t >= END_TIME: break
      m = re.search('EXIT STATE (.*?)$', l)
      # print t-lasttime, m.group(1)
      ax1.add_patch(patches.Rectangle((0.3 + 0.3*(worker+1), lasttime), 0.2, t-lasttime, color=colors[m.group(1)]))

# Print figure
print "Generating " + out_name + '.png'
fig1.set_size_inches(5, 1000)
fig1.savefig(out_name + '.png', dpi=30, bbox_inches='tight')

