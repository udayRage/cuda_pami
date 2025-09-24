#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib> // For atoi and getenv
#include <algorithm> // For reverse

static std::random_device rd;
static std::mt19937 gen(rd());

// Function to generate a random item number
int generateRandomItem(int maxItems) {
    std::uniform_int_distribution<> dis(1, maxItems);
    return dis(gen);
}

// Function to create a square-shaped file
void generateSquareFile(const std::string& fileName, size_t fileSize, char delimiter, int maxItems) {
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    size_t lineSize = static_cast<size_t>(std::sqrt(fileSize) + 0.5); // Rounded size per line
    size_t remainingSize = fileSize;

    while (remainingSize > 0) {
        std::string line;
        size_t lineItemSize = lineSize / (sizeof(int) + 1); // Approximate items per line

        for (size_t i = 0; i < lineItemSize; ++i) {
            line += std::to_string(generateRandomItem(maxItems));
            if (i < lineItemSize - 1) line += delimiter;
        }

        line += '\n';

        if (line.size() > remainingSize) break;

        file << line;
        remainingSize -= line.size();
    }

    file.close();
    std::cout << "Square-shaped file created: " << fileName << std::endl;
}

// Function to create a triangular-shaped file
// Function to create a triangular-shaped file
void generateTriangleFile(const std::string& fileName, size_t fileSize, char delimiter, int maxItems) {
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    size_t totalBytes = 0;
    size_t currentLine = 1;
    std::vector<size_t> lines; // To store the line counts for descending order

    // First pass: Ascending order to determine line counts
    while (totalBytes < fileSize) {
        std::string line;
        for (size_t i = 0; i < currentLine; ++i) {
            line += std::to_string(generateRandomItem(maxItems));
            if (i < currentLine - 1) line += delimiter;
        }

        line += '\n';

        if (totalBytes + line.size() > fileSize) break;

        lines.push_back(currentLine); // Store the number of items for this line
        totalBytes += line.size();
        ++currentLine;
    }

    // Reset and start the second pass: Descending order
    totalBytes = 0;
    std::reverse(lines.begin(), lines.end()); // Reverse the line counts for descending order

    for (size_t items : lines) {
        std::string line;
        for (size_t i = 0; i < items; ++i) {
            line += std::to_string(generateRandomItem(maxItems));
            if (i < items - 1) line += delimiter;
        }

        line += '\n';

        if (totalBytes + line.size() > fileSize) break;

        file << line;
        totalBytes += line.size();
    }

    file.close();
    std::cout << "Triangular-shaped file created: " << fileName << std::endl;
}



int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <fileName> <fileSize> <delimiter> <shapeChoice> optional:<maxItems>" << std::endl;
        return 1;
    }

    // int maxItems = max of int16 
    int maxItems = INT32_MAX;

    if (argc == 6) {
        // if it says INT_32_MAX, it means that the user wants to set the maxItems
        // if it says INT_16_MAX, it means that the user wants to set the maxItems to the max of int16
        // if it says INT_8_MAX, it means that the user wants to set the maxItems to the max of int8
        if (std::string(argv[5]) == "INT_32_MAX") {
            maxItems = INT32_MAX;
        } else if (std::string(argv[5]) == "INT_16_MAX") {
            maxItems = INT16_MAX;
        } else if (std::string(argv[5]) == "INT_8_MAX") {
            maxItems = INT8_MAX;
        } else {
            maxItems = std::atoi(argv[5]);
        }
    }

    std::string fileName = argv[1];
    std::string sizeInput = argv[2];
    char delimiter = argv[3][0];
    int shapeChoice = std::atoi(argv[4]);

    size_t fileSize;
    if (sizeInput.back() == 'M') {
        fileSize = std::stoull(sizeInput.substr(0, sizeInput.size() - 1)) * 1024 * 1024;
    } else if (sizeInput.back() == 'G') {
        fileSize = std::stoull(sizeInput.substr(0, sizeInput.size() - 1)) * 1024 * 1024 * 1024;
    } else {
        std::cerr << "Invalid size input." << std::endl;
        return 1;
    }

    if (shapeChoice == 1) {
        generateSquareFile(fileName, fileSize, delimiter, maxItems);
    } else if (shapeChoice == 2) {
        generateTriangleFile(fileName, fileSize, delimiter, maxItems);
    } else {
        std::cerr << "Invalid choice." << std::endl;
        return 1;
    }

    return 0;
}

// g++ -std=c++11 -O3 -o file_generator file_generator.cpp