#include <iostream>

#include <unistd.h>

//#include <likwid.h>
#include <likwid-marker.h>



int main() {
    LIKWID_MARKER_INIT;

    LIKWID_MARKER_START("measure main");

    std::cout << "Hello World!";
    sleep(1);

    LIKWID_MARKER_STOP("measure main");
    LIKWID_MARKER_CLOSE;

    return 0;
}
