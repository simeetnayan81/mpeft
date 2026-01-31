#ifndef MEMORY_MAP_HPP
#define MEMORY_MAP_HPP

#include <string>

class MemoryMap {
public:
    MemoryMap(const std::string& path);
    ~MemoryMap();

    // Deleted copy constructor/assignment to prevent double-unmapping
    MemoryMap(const MemoryMap&) = delete;
    MemoryMap& operator=(const MemoryMap&) = delete;

    void* data() const { return addr_; }
    size_t size() const { return size_; }

private:
    void* addr_;
    size_t size_;
    int fd_;
};

#endif