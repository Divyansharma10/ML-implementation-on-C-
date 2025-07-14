all: simple_classification.exe

simple_classification.exe: examples/simple_classification.cpp algorithms/mlp.cpp
    g++ -o simple_classification.exe examples/simple_classification.cpp algorithms/mlp.cpp -Ialgorithms

clean:
    rm -f *.exe