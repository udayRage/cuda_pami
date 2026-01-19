#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <omp.h> // OpenMP Header
#include <unistd.h> 

// Compile
// g++ ffiminer.cpp -fopenmp -O3 -o ffiminer 

// Data Structures
struct Element {
    int tid;
    float value;
};

struct PatternNode {
    std::vector<int> items;
    std::vector<Element> elements;
    float support;
};

class FFIMiner {
private:
    std::string iFile;
    std::string oFile;
    float minSupInput;
    double minSupThreshold;
    std::string sep;
    
    // Parallel Options
    int numThreads;
    std::string scheduleType; // "static" or "dynamic"

    int dbLen = 0;
    
    std::map<std::string, int> strToInt;
    std::map<int, std::string> intToStr;
    int itemCounter = 0;

    std::vector<std::pair<std::string, float>> finalPatterns;

    double startTime;
    double endTime;
    size_t peakMemory = 0;

    std::vector<std::string> split(const std::string &s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

    double getTime() {
        using namespace std::chrono;
        return duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
    }

    size_t getMemoryRSS() {
        long rss = 0L;
        FILE* fp = NULL;
        if ((fp = fopen( "/proc/self/statm", "r" )) == NULL)
            return (size_t)0L;
        if (fscanf(fp, "%*s%ld", &rss) != 1) {
            fclose(fp);
            return (size_t)0L;
        }
        fclose(fp);
        return (size_t)rss * (size_t)sysconf( _SC_PAGESIZE);
    }

public:
    FFIMiner(std::string inputFile, float minSup, std::string separator, int threads, std::string sched) 
        : iFile(inputFile), minSupInput(minSup), sep(separator), numThreads(threads), scheduleType(sched) {}

    void mine() {
        startTime = getTime();
        
        // --- Phase 1: Sequential Reading (I/O is rarely the bottleneck) ---
        std::ifstream file(iFile);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << iFile << std::endl;
            return;
        }

        std::string line;
        std::map<int, std::vector<Element>> initialItems; 
        char sepChar;
        if (sep == "tab") sepChar = '\t';
        else if (sep == "space") sepChar = ' ';
        else sepChar = sep[0];

        // char sepChar = sep[0]; 
        std::cout << "Separator set to '" << sepChar << "'" << std::endl;
        std::cout << "Separator string: " << sep << std::endl;
        // if (sep == "\\t") sepChar = '\t';

        int lineNo = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            size_t colonPos = line.find(':');
            if (colonPos == std::string::npos) continue;

            std::string itemPart = line.substr(0, colonPos);
            std::string quantPart = line.substr(colonPos + 1);

            std::vector<std::string> tItems = split(itemPart, sepChar);
            std::vector<std::string> tQuants = split(quantPart, sepChar);

            if (tItems.size() != tQuants.size()) {
                lineNo++;
                continue;
            }

            for (size_t i = 0; i < tItems.size(); ++i) {
                std::string itemStr = tItems[i];
                itemStr.erase(0, itemStr.find_first_not_of(" \t\r\n"));
                itemStr.erase(itemStr.find_last_not_of(" \t\r\n") + 1);
                if (itemStr.empty()) continue;

                float fuzzyVal = std::stof(tQuants[i]);

                if (strToInt.find(itemStr) == strToInt.end()) {
                    strToInt[itemStr] = itemCounter;
                    intToStr[itemCounter] = itemStr;
                    itemCounter++;
                }
                int itemId = strToInt[itemStr];
                initialItems[itemId].push_back({lineNo, fuzzyVal});
            }
            lineNo++;
        }
        dbLen = lineNo;
        file.close();

        if (minSupInput < 1.0) minSupThreshold = minSupInput * dbLen;
        else minSupThreshold = minSupInput;

        std::vector<PatternNode> sortedFrequentItems;
        for (auto& pair : initialItems) {
            float sum = 0;
            for (const auto& el : pair.second) sum += el.value;
            if (sum >= minSupThreshold) {
                PatternNode node;
                node.items.push_back(pair.first);
                node.elements = pair.second;
                node.support = sum;
                sortedFrequentItems.push_back(node);
                finalPatterns.push_back({intToStr[pair.first], sum});
            }
        }

        std::sort(sortedFrequentItems.begin(), sortedFrequentItems.end(), 
            [](const PatternNode& a, const PatternNode& b) {
                return a.support > b.support;
            });

        // --- Phase 2: Parallel DFS ---
        
        // Set OpenMP Settings
        omp_set_num_threads(numThreads);
        
        // Map string input to OpenMP schedule constants
        omp_sched_t method = omp_sched_static;
        int chunk_size = 0; // 0 lets OpenMP decide optimal default
        
        if (scheduleType == "dynamic") {
            method = omp_sched_dynamic;
            chunk_size = 1; // Granularity: 1 iteration per fetch (good for unbalanced trees)
        }

        omp_set_schedule(method, chunk_size);

        std::cout << "Starting Parallel Mining with " << numThreads << " threads using " 
                  << scheduleType << " schedule." << std::endl;

        #pragma omp parallel
        {
            // Thread-local storage for patterns to avoid locking on every push_back
            std::vector<std::pair<std::string, float>> localPatterns;

            // OpenMP Runtime Schedule lets us control it via omp_set_schedule above
            #pragma omp for schedule(runtime)
            for (size_t i = 0; i < sortedFrequentItems.size(); ++i) {
                std::vector<PatternNode> newCandidates;
                
                // Inner Loop (Intersection) - Serial within the thread
                for (size_t j = i + 1; j < sortedFrequentItems.size(); ++j) {
                    PatternNode& p1 = sortedFrequentItems[i];
                    PatternNode& p2 = sortedFrequentItems[j];
                    
                    std::vector<Element> newElements;
                    float newSupport = 0.0;
                    size_t idx1 = 0, idx2 = 0;

                    // Compute Intersection
                    while (idx1 < p1.elements.size() && idx2 < p2.elements.size()) {
                        if (p1.elements[idx1].tid == p2.elements[idx2].tid) {
                            float minVal = std::min(p1.elements[idx1].value, p2.elements[idx2].value);
                            newElements.push_back({p1.elements[idx1].tid, minVal});
                            newSupport += minVal;
                            idx1++; idx2++;
                        } else if (p1.elements[idx1].tid < p2.elements[idx2].tid) {
                            idx1++;
                        } else {
                            idx2++;
                        }
                    }

                    if (newSupport >= minSupThreshold) {
                        PatternNode newNode;
                        newNode.items = p1.items;
                        newNode.items.push_back(p2.items.back());
                        newNode.elements = std::move(newElements);
                        newNode.support = newSupport;

                        // Save pattern to thread local storage
                        std::string patternStr = "";
                        for(size_t k=0; k<newNode.items.size(); ++k) {
                            patternStr += intToStr[newNode.items[k]];
                            if(k < newNode.items.size() - 1) patternStr += sep;
                        }
                        localPatterns.push_back({patternStr, newSupport});

                        newCandidates.push_back(std::move(newNode));
                    }
                }

                // If this branch has children, recurse (using Serial DFS for the subtree)
                if (!newCandidates.empty()) {
                    dfs_serial(newCandidates, localPatterns);
                }
            }

            // Merge local results into global results
            #pragma omp critical
            {
                finalPatterns.insert(finalPatterns.end(), localPatterns.begin(), localPatterns.end());
            }
        } // End Parallel Region

        endTime = getTime();
        peakMemory = getMemoryRSS();
    }

    // Recursive DFS (Executed serially by a single thread for its assigned subtree)
    void dfs_serial(std::vector<PatternNode>& candidates, std::vector<std::pair<std::string, float>>& outputStore) {
        
        for (size_t i = 0; i < candidates.size(); ++i) {
            std::vector<PatternNode> newCandidates;
            
            for (size_t j = i + 1; j < candidates.size(); ++j) {
                PatternNode& p1 = candidates[i];
                PatternNode& p2 = candidates[j];
                
                std::vector<Element> newElements;
                float newSupport = 0.0;
                size_t idx1 = 0, idx2 = 0;

                while (idx1 < p1.elements.size() && idx2 < p2.elements.size()) {
                    if (p1.elements[idx1].tid == p2.elements[idx2].tid) {
                        float minVal = std::min(p1.elements[idx1].value, p2.elements[idx2].value);
                        newElements.push_back({p1.elements[idx1].tid, minVal});
                        newSupport += minVal;
                        idx1++; idx2++;
                    } else if (p1.elements[idx1].tid < p2.elements[idx2].tid) {
                        idx1++;
                    } else {
                        idx2++;
                    }
                }

                if (newSupport >= minSupThreshold) {
                    PatternNode newNode;
                    newNode.items = p1.items;
                    newNode.items.push_back(p2.items.back());
                    newNode.elements = std::move(newElements);
                    newNode.support = newSupport;

                    std::string patternStr = "";
                    for(size_t k=0; k<newNode.items.size(); ++k) {
                        patternStr += intToStr[newNode.items[k]];
                        if(k < newNode.items.size() - 1) patternStr += sep;
                    }
                    outputStore.push_back({patternStr, newSupport});

                    newCandidates.push_back(std::move(newNode));
                }
            }

            if (!newCandidates.empty()) {
                dfs_serial(newCandidates, outputStore);
            }
        }
    }

    void save(std::string outputFileName) {
        // Sort final results for consistency
        std::sort(finalPatterns.begin(), finalPatterns.end(), 
            [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
                 if (a.first.length() != b.first.length())
                    return a.first.length() < b.first.length();
                 return a.first < b.first;
            });

        std::ofstream out(outputFileName);
        for (const auto& p : finalPatterns) {
            out << p.first << ":" << p.second << "\n";
        }
        out.close();
    }

    void printResults() {
        std::cout << "\n--- FFIMiner Results ---" << std::endl;
        std::cout << "Execution Time: " << std::fixed << std::setprecision(4) << (endTime - startTime) << " seconds" << std::endl;
        std::cout << "Peak Memory: " << (peakMemory / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Patterns Found: " << finalPatterns.size() << std::endl;
        std::cout << "----------------------------" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // Expected args: inputFile minSup outputFile sep numThreads scheduleType
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <inputFile> <minSup> [outputFile] [sep] [numThreads] [schedule]" << std::endl;
        std::cout << "Defaults: output.txt, tab, 1 thread, static" << std::endl;
        return 1;
    }

    std::string iFile = argv[1];
    float minSup = std::stof(argv[2]);
    std::string oFile = "patterns.txt";
    std::string sep = "\t";
    int threads = 1;
    std::string schedule = "static";

    if (argc >= 4) oFile = argv[3];
    if (argc >= 5) sep = argv[4];
    if (argc >= 6) threads = std::stoi(argv[5]);
    if (argc >= 7) schedule = argv[6];

    FFIMiner miner(iFile, minSup, sep, threads, schedule);
    miner.mine();
    miner.printResults();
    miner.save(oFile);

    return 0;
}