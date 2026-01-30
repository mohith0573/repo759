#include <iostream>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    // 1. Check if the argument N is provided
    if (argc < 2) {
        // Ideally print usage, but for this homework we just exit
        return 1;
    }

    // 2. Convert the command line argument (string) to an integer
    int N = std::atoi(argv[1]);

    // 3. Task (b): Print 0 to N using printf
    for (int i = 0; i <= N; ++i) {
        printf("%d ", i);
    }
    printf("\n"); // Newline at the end

    // 4. Task (c): Print N to 0 using std::cout
    for (int i = N; i >= 0; --i) {
        std::cout << i << " ";
    }
    std::cout << "\n"; // Newline at the end

    return 0;
}
