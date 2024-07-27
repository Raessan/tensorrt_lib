#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Read data from file and store the result in std::vector<std::vector<float>>
std::vector<std::vector<float>>  read_file(const std::string& filename) {
    std::vector<std::vector<float>> result;
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::vector<float> numbers;
        float number;

        while (ss >> number) {
            numbers.push_back(number);
            if (ss.peek() == ',') ss.ignore();
        }

        std::vector<float> partial_result(numbers.size());
        for (size_t i = 0; i < numbers.size(); ++i) {
            partial_result[i] = numbers[i];
        }

        // Add the float array to the result vector
        result.push_back(partial_result);
    }

    return result;
}

// Function to calculate MAE for two arrays of floats
float calculate_mae(std::vector<float> matrix1, std::vector<float> matrix2, size_t size) {
    float mae = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        mae += std::abs(matrix1[i] - matrix2[i]);
    }

    return mae / size;
}

// Function to calculate MAE for two std::vector<std::vector<float>> arrays
float calculate_mae(const std::vector<std::vector<float>>& matrix1, const std::vector<std::vector<float>>& matrix2, size_t size) {
    float mae = 0.0f;
    size_t numRows = matrix1.size();

    for (size_t row = 0; row < numRows; ++row) {
        for (size_t col = 0; col < size; ++col) {
            mae += std::abs(matrix1[row][col] - matrix2[row][col]);
        }
    }

    return mae / (numRows * size);
}

// Function to check if there's an error in cuda-related functions
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}