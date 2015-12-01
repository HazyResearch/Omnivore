#ifndef _TASK_QUEUE_H
#define _TASK_QUEUE_H

#include <mutex>
#include <vector>
#include <thread>
#include <functional>

class Task {
public:

  std::mutex * mymutex;

  std::vector<std::mutex*> depend_ons; 

  std::function<void()> func;

  Task(std::function<void()> _func){
    func = _func;
    mymutex = new std::mutex();
  }

};

void run_task(Task * task){
  for(auto pmutex : task->depend_ons){
    pmutex->lock();
    pmutex->unlock();
  }
  task->func();
  task->mymutex->unlock();
}

class TaskQueue {
public:

  std::vector<Task> tasks;

  void prepare(){
    // unlock everything
    for(auto task : tasks){
      task.mymutex->unlock();
      for(auto & pmutex : task.depend_ons){
        pmutex->unlock();
      }
    }
    // lock itself
    for(auto task : tasks){
      task.mymutex->lock();
    }
  }

  void run(){

    std::vector<std::thread> threads;

    for(size_t i=0;i<tasks.size();i++){
      threads.push_back(std::thread(run_task, &tasks[i]));
    }

    for(auto & thread : threads){
      thread.join();
    }

  }

};


#endif