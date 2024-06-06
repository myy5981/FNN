#include <SDL2/SDL.h>

#include "network.h"

SDL_Window* window;
SDL_Renderer* renderer;
float bmp[28 * 28];

#define TIMES 3
fnn_t* fnn;

void test(int time) {
    FILE* t10k_images = fopen("./mnist/t10k-images.idx3-ubyte", "rb");
    FILE* t10k_lables = fopen("./mnist/t10k-labels.idx1-ubyte", "rb");
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

void train() {
    fnn = new_fnn(0.01, CORSS_ENTROPY);
    fnn_add_layer(fnn, 28 * 28, 64, RELU);
    fnn_add_layer(fnn, 64, 64, RELU);
    fnn_add_layer(fnn, 64, 64, RELU);
    fnn_add_layer(fnn, 64, 10, SOFTMAX);

    FILE* train_images = fopen("./mnist/train-images.idx3-ubyte", "rb");
    FILE* train_lables = fopen("./mnist/train-labels.idx1-ubyte", "rb");
    if (train_images == NULL || train_lables == NULL) {
        perror("fopen");
        exit(1);
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
    return;
}
int bmp_is_not_blank() {
    for (int i = 0; i < 28 * 28; i++) {
        if (bmp[i] != 0.0f) {
            return 1;
        }
    }
    return 0;
}

void init() {
    for (int i = 0; i < 28 * 28; i++) {
        bmp[i] = 0.0f;
    }
    SDL_Init(SDL_INIT_VIDEO);

    window = SDL_CreateWindow("SDL Rectangle", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 280, 280, SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
}

void reset() {
    memset(bmp, 0, sizeof(float) * 28 * 28);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
}

void draw(int bx, int by) {
    if (bx >= 0 && bx < 280 && by >= 0 && by < 280) {
        int x = bx / 10;
        int y = by / 10;
        if (bmp[y * 28 + x] == 0) {
            bmp[y * 28 + x] = 1;
            SDL_Rect rect = {10 * x, 10 * y, 10, 10};
            SDL_RenderFillRect(renderer, &rect);
            SDL_RenderPresent(renderer);
        }
    }
}

int main() {
    train();
    printf(
        "usage: "
        "按住或点击鼠标左键绘制，点击鼠标右键重置，点击鼠标中键启动推理，推理结"
        "果显示在控制台。\n");

    init();


    int quit = 0;
    SDL_Event event;
    int mouse1_down = 0;

    while (!quit) {
        while (SDL_PollEvent(&event) != 0) {
            if (event.type == SDL_QUIT) {
                quit = 1;
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == 1) {
                    mouse1_down = 1;
                    draw(event.button.x, event.button.y);
                } else if (event.button.button == 3) {
                    mouse1_down = 0;
                    reset();
                } else if (event.button.button == 2) {
                    if (bmp_is_not_blank()) {
                        printf("waiting...\n");
                        FVECTOR out = fnn_forward(fnn,bmp);
                        float max = 0;int res;
                        for(int i = 0;i < fnn->tail->output_num;i++){
                            if(out[i]>max){
                                res = i;
                                max = out[i];
                            }
                            printf("probability of number %d:%f\n",i,out[i]);
                        }
                        printf("预测结果：%d\n",res);
                    }
                }
            } else if (event.type == SDL_MOUSEMOTION) {
                if (mouse1_down) {
                    draw(event.motion.x, event.motion.y);
                }
            } else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == 1) {
                    mouse1_down = 0;
                }
            }
        }
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    fnn_destory(fnn);
    fnn = NULL;

    return 0;
}
