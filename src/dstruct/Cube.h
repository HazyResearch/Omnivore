#ifndef _CUBE_H
#define _CUBE_H

#include <string>

using namespace std;

class Cube {
public:

  string name;
  int N; // # rows = # cols
  int I; // # depths
  int B; // # batches

  Cube(){
    
  }

  Cube(string _name, int _N, int _I, int _B){
    name = _name;
    N = _N;
    I = _I;
    B = _B;
  }

};

#endif