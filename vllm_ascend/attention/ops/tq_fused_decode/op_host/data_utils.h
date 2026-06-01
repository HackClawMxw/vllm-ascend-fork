#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

inline bool ReadFile(const char* filePath, void* buffer, uint64_t size) {
    FILE* fp = fopen(filePath, "rb");
    if (fp == nullptr) return false;
    uint64_t readSize = fread(buffer, 1, size, fp);
    fclose(fp);
    return readSize == size;
}

inline bool WriteFile(const char* filePath, const void* buffer, uint64_t size) {
    FILE* fp = fopen(filePath, "wb");
    if (fp == nullptr) return false;
    uint64_t writeSize = fwrite(buffer, 1, size, fp);
    fclose(fp);
    return writeSize == size;
}

#endif
