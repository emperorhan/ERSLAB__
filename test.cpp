#include <cstdio>
#include <iostream>
#include <algorithm>
#include <random>
#include <ctime>
using namespace std;
default_random_engine engine(static_cast<unsigned int>(time(0)));
uniform_int_distribution<unsigned int> Video_Length_GEN(0, 3600);


int main(){
	for(int video = 0; video < 1000; video++){
		int rnd = Video_Length_GEN(engine);
        rnd -= rnd%(60);
        int length = 3600 + rnd;
        printf("%d, ", length);
	}
}