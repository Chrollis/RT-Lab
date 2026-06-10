#include <graphics.h>

int main() {
    ExMessage msg;
    initgraph(800, 600);
    bool running = true;

    while (running) {
        if (peekmessage(&msg)) {
        }

        BeginBatchDraw();
        cleardevice();
        EndBatchDraw();
        Sleep(100);
    }
    closegraph();
    return 0;
}