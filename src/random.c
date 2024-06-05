#include "random.h"

#include <sys/random.h>
#include <stdlib.h>
#include <math.h>

static float randf(){
    unsigned char byte;
    retry:
    if(getrandom(&byte,1,0)!=1){
        exit(1);
    }
    if(byte==0){
        goto retry;
    }
    return ((float)byte) / 255.0f;
}

static float gaussian_randf(){
    float v1 = randf();
    float v2 = randf();
    float x = sqrtf(-2.0f * logf(v1)) * sinf(2.0f * 3.1415926f * v2);
    return x;
}

float randomf(float mean, float stddev){
    return mean + gaussian_randf() * stddev;
}
