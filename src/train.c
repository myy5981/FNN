#include "network.h"
#include "loss.h"
#include "activation.h"

#include <stdio.h>
#include <stdlib.h>

int main(void){
    fnn_t* fnn = new_fnn(0.01,MEAN_SQUARE);
    fnn_add_layer(fnn,28*28,64,RELU);
    fnn_add_layer(fnn,64,64,RELU);
    fnn_add_layer(fnn,64,64,RELU);
    fnn_add_layer(fnn,64,10,SOFTMAX);
    
    FILE* train_images = fopen("./mnist/train-images.idx3-ubyte","rb");
    FILE* train_lables = fopen("./mnist/train-labels.idx1-ubyte","rb");
    if(train_images==NULL||train_lables==NULL){
        perror("fopen");
        return 0;
    }
    fseek(train_images,16,SEEK_SET);
    fseek(train_lables,8,SEEK_SET);
    unsigned char buf[28*28];
    float input[28*28];
    float real[10];
    for (int i = 0; i < 60000; i++) {
        if(fread(buf,1,28*28,train_images)!=28*28){
            perror("fread");
            exit(1);
        }
        for (int j = 0; j < 28*28; j++) {
            input[j]=((float)buf[j])/255.0f;
        }
        if(fread(buf,1,1,train_lables)!=1){
            perror("fread");
            exit(1);
        }
        for(int j = 0;j<10;j++) {
            if(j==buf[0]){
                real[j]=1.0f;
            }else{
                real[j]=0.0f;
            }
        }
        fnn_forward(fnn,input);
        fnn_backward(fnn,real);
        fnn_forward(fnn,input);
        fnn_backward(fnn,real);
        fnn_forward(fnn,input);
        fnn_backward(fnn,real);
        if((i+1)%1000==0){
            printf("loss after %d iteration:%f\n",i+1,fnn->loss->loss_func(fnn->tail->output,real,fnn->tail->output_num));
        }
    }
    return 0;
}