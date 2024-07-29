#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Read data from file and store the result in std::vector<std::vector<float>>
// If num_lines <= 0, the whole document is read.
std::vector<std::vector<float>> read_file(const std::string& filename, int num_lines=0) {
    std::vector<std::vector<float>> result;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open file.");
    }

    std::string line;
    int lines_read = 0;

    while (std::getline(infile, line)) {
        if (num_lines > 0 && lines_read >= num_lines) {
            break; // Stop reading if we have read the specified number of lines
        }

        std::stringstream ss(line);
        std::vector<float> numbers;
        float number;

        while (ss >> number) {
            numbers.push_back(number);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }

        result.push_back(numbers);
        ++lines_read;
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