#include "activation.h"
#include "loss.h"
#include "network.h"

#include <stdio.h>
#include <stdlib.h>

#define TIMES 3
fnn_t* fnn;

void test(int time) {
    FILE* t10k_images = fopen("./MNIST/raw/t10k-images-idx3-ubyte", "rb");
    FILE* t10k_lables = fopen("./MNIST/raw/t10k-labels-idx1-ubyte", "rb");
    if (t10k_images == NULL || t10k_lables == NULL) {
        perror("fopen");
        exit(1);
    }
    unsigned char buf[28 * 28];
    float input[28 * 28];
    fseek(t10k_images, 16, SEEK_SET);
    fseek(t10k_lables, 8, SEEK_SET);
    int correct = 0;
    for (int i = 0; i < 10000; i++) {
        if (fread(buf, 1, 28 * 28, t10k_images) != 28 * 28) {
            perror("fread");
            exit(1);
        }
        for (int j = 0; j < 28 * 28; j++) {
            input[j] = ((float)buf[j]) / 255.0f;
        }
        if (fread(buf, 1, 1, t10k_lables) != 1) {
            perror("fread");
            exit(1);
        }
        FVECTOR out = fnn_forward(fnn, input);
        float max = 0;
        int res = 0;
        for (int i = 0; i < fnn->tail->output_num; i++) {
            if (out[i] > max) {
                res = i;
                max = out[i];
            }
        }
        if (res == buf[0]) {
            correct++;
        }
    }
    printf("第%d轮训练后测试集准确率: %f%%\n", time, ((float)correct) / 100.0f);
    fclose(t10k_images);
    fclose(t10k_lables);
}

int main(void) {
    fnn = new_fnn(0.01, CORSS_ENTROPY);
    fnn_add_layer(fnn, 28 * 28, 64, RELU);
    fnn_add_layer(fnn, 64, 64, RELU);
    fnn_add_layer(fnn, 64, 64, RELU);
    fnn_add_layer(fnn, 64, 10, SOFTMAX);

    FILE* train_images = fopen("./MNIST/raw/train-images-idx3-ubyte", "rb");
    FILE* train_lables = fopen("./MNIST/raw/train-labels-idx1-ubyte", "rb");
    if (train_images == NULL || train_lables == NULL) {
        perror("fopen");
        return 0;
    }
    unsigned char buf[28 * 28];
    float input[28 * 28];
    float real[10];
    test(0);
    for (int x = 0; x < TIMES; x++) {
        fseek(train_images, 16, SEEK_SET);
        fseek(train_lables, 8, SEEK_SET);
        for (int i = 0; i < 60000; i++) {
            if (fread(buf, 1, 28 * 28, train_images) != 28 * 28) {
                perror("fread");
                exit(1);
            }
            for (int j = 0; j < 28 * 28; j++) {
                input[j] = ((float)buf[j]) / 255.0f;
            }
            if (fread(buf, 1, 1, train_lables) != 1) {
                perror("fread");
                exit(1);
            }
            for (int j = 0; j < 10; j++) {
                if (j == buf[0]) {
                    real[j] = 1.0f;
                } else {
                    real[j] = 0.0f;
                }
            }
            fnn_forward(fnn, input);
            fnn_backward(fnn, real);
            if ((i + 1) % 10000 == 0) {
                printf("第 %d 次迭代损失值:%f\n", i + 1,
                       fnn->loss->loss_func(fnn->tail->output, real, fnn->tail->output_num));
            }
        }
        test(x + 1);
    }
    fnn_destory(fnn);
    fnn = NULL;
    return 0;
}