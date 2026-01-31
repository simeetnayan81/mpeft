#include "MemoryMap.hpp" // If using include_directories(include) in CMake
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

MemoryMap::MemoryMap(const std::string& path) : addr_(nullptr), size_(0), fd_(-1) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ == -1) throw std::runtime_error("Could not open file: " + path);

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        throw std::runtime_error("Could not get file size.");
    }
    size_ = sb.st_size;

    addr_ = mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (addr_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed.");
    }
}

MemoryMap::~MemoryMap() {
    if (addr_ != nullptr && addr_ != MAP_FAILED) {
        munmap(addr_, size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}