
import sys, os
import datetime
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.set_ylim([6498109,7898109])
ax1.set_xlim([0,0.6+0.3*18])
ax1.yaxis.grid(True)
ax1.yaxis.set_ticks([i*10.0 for i in range(649810,789810)])

def fsec(s):
  hours, minutes, seconds = [float(val) for val in s.split(':')]
  return int((hours*3600+minutes*60+seconds)*1000)

colors = {
  1: "red",
  2: "blue",
  3: "pink",
  4: "yellow",
  5: "black"
}

lasttime = None
for l in open('log_titan/convmodel.log'):
  l = l.rstrip()
  ss = l.split(' ')
  if len(ss) < 2: continue
  try: t = fsec(ss[1]) 
  except: continue

  if t > 7898109: break

  if 'ENTER' in l and 'IDLE' not in l:
    lasttime = t
  if 'EXIT' in l and 'IDLE' not in l:
    if lasttime != None:
      m = re.search('EXIT STATE (.*?)$', l)
      print lasttime, t, m.group(1)
      ax1.add_patch(patches.Rectangle((0.1, lasttime), 0.2, t-lasttime, color=colors[int(m.group(1))]))

lasttime = None
for l in open('log_titan/fcmodel.log'):
  l = l.rstrip()
  ss = l.split(' ')
  if len(ss) < 2: continue
  try: t = fsec(ss[1]) 
  except: continue

  if t > 7898109: break

  if 'ENTER' in l and 'IDLE' not in l:
    lasttime = t
  if 'EXIT' in l and 'IDLE' not in l:
    if lasttime != None:
      m = re.search('EXIT STATE (.*?)$', l)
      print lasttime, t, m.group(1)
      ax1.add_patch(patches.Rectangle((0.3, lasttime), 0.2, t-lasttime, color=colors[int(m.group(1))]))


for worker in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
  lasttime = None
  for l in open('log_titan/worker%d.log' % worker):
    l = l.rstrip()
    ss = l.split(' ')
    if len(ss) < 2: continue
    try: t = fsec(ss[1]) 
    except: continue

    if t > 7898109: break

    if 'ENTER' in l and 'IDLE' not in l:
      lasttime = t
    if 'EXIT' in l and 'IDLE' not in l:
      if lasttime != None:
        m = re.search('EXIT STATE (.*?)$', l)
        if m:
          print lasttime, t, m.group(1)
          ax1.add_patch(patches.Rectangle((0.3 + 0.3*worker, lasttime), 0.2, t-lasttime, color=colors[int(m.group(1))]))


print "PRINTING FIGURES"
fig1.set_size_inches(5, 1000)
fig1.savefig('rect5.png', dpi=90, bbox_inches='tight')









