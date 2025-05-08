#include "load_files.h"

#include <iostream>
#include <system_error>
#include <stdexcept>

#if _WIN32
# define WIN32_LEAN_AND_MEAN
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# include <windows.h>

#define PATH_PREFIX "G:\\vectors\\siftsmall\\siftsmall"

struct FileMappingData {
    HANDLE hFileMapping;
    void *mapping_ptr;
};

static void *map_file(const char *fname, FileMappingData *data, size_t *filesize);
static void unmap_file(FileMappingData *data);
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define PATH_PREFIX "/mnt/g/vectors/sift/sift"

struct FileMappingData {
    void *mapping_ptr;
};

static void *map_file(const char *fname, FileMappingData *data, size_t *filesize);
static void unmap_file(FileMappingData *data);
#endif

RawVectorData data_base;
RawVectorData data_query;
RawVectorData data_learn;
IntVectorData data_ground;

static FileMappingData mapping_data[4];

void load_vector(const char *fname, RawVectorData *data, FileMappingData *map) {
    data->filestart = map_file(fname, map, &data->filesize);
    data->vec = (float *) data->filestart;
    data->dim = *(int *) data->filestart;
    data->length = data->filesize / 4 / (1 + data->dim);
}

void load_vector(const char *fname, IntVectorData *data, FileMappingData *map) {
    data->filestart = map_file(fname, map, &data->filesize);
    data->vec = (int *) data->filestart;
    data->dim = *(int *) data->filestart;
    data->length = data->filesize / 4 / (1 + data->dim);
}

void load_files() {
    load_vector(PATH_PREFIX "_base.fvecs", &data_base, &mapping_data[0]);
    try {
        load_vector(PATH_PREFIX "_query.fvecs", &data_query, &mapping_data[1]);
        try {
            load_vector(PATH_PREFIX "_learn.fvecs", &data_learn, &mapping_data[2]);
            try {
                load_vector(PATH_PREFIX "_groundtruth.ivecs", &data_ground, &mapping_data[3]);
            } catch (...) {
                unmap_file(&mapping_data[2]);
                throw;
            }
        } catch (...) {
            unmap_file(&mapping_data[1]);
            throw;
        }
    } catch (...) {
        unmap_file(&mapping_data[0]);
        throw;
    }
}

void close_files() {
    unmap_file(&mapping_data[3]);
    unmap_file(&mapping_data[2]);
    unmap_file(&mapping_data[1]);
    unmap_file(&mapping_data[0]);
}

#if _WIN32
static void *map_file(const char *fname, FileMappingData *data, size_t *filesize) {
    HANDLE hFile = CreateFileA(
        fname, /* lpFileName */
        GENERIC_READ, /* dwDesiredAccess */
        FILE_SHARE_READ, /* dwShareMode */
        nullptr, /* lpSecurityAttributes */
        OPEN_EXISTING, /* dwCreationDisposition */
        0, /* dwFlagsAndAttributes */
        nullptr /* hTemplateFile */
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        DWORD error = GetLastError();
        std::cerr << "error opening file '" << fname << "'" << std::endl;
        throw std::system_error(error, std::system_category());
    }
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) {
        DWORD error = GetLastError();
        std::cerr << "error getting size of file '" << fname << "'" << std::endl;
        CloseHandle(hFile);
        throw std::system_error(error, std::system_category());
    }
    if (filesize) *filesize = file_size.QuadPart;
    data->hFileMapping = CreateFileMappingA(
        hFile, /* hFile */
        nullptr, /* lpFileMappingAttributes */
        PAGE_READONLY, /* flProtect */
        0, /* dwMaximumSizeHigh */
        0, /* dwMaximumSizeLow */
        nullptr /* lpName */
    );
    if (data->hFileMapping == nullptr) {
        DWORD error = GetLastError();
        std::cerr << "error creating mapping for file '" << fname << "'" << std::endl;
        CloseHandle(hFile);
        throw std::system_error(error, std::system_category());
    }
    CloseHandle(hFile);
    data->mapping_ptr = MapViewOfFile(
        data->hFileMapping, /* hFileMappingObject */
        FILE_MAP_READ, /* dwDesiredAccess */
        0, /* dwFileOffsetHigh */
        0, /* dwFileOffsetLow */
        0 /* dwNumberOfBytesToMap */
    );
    if (data->mapping_ptr == nullptr) {
        DWORD error = GetLastError();
        std::cerr << "error creating mapping for file '" << fname << "'" << std::endl;
        CloseHandle(data->hFileMapping);
        throw std::system_error(error, std::system_category());
    }
    return data->mapping_ptr;
}

static void unmap_file(FileMappingData *data) {
    UnmapViewOfFile(data->mapping_ptr);
    CloseHandle(data->hFileMapping);
}

#else

static void *map_file(const char *fname, FileMappingData *data, size_t *filesize) {
    int fd = open(fname, O_RDONLY);
    if (fd == -1) {
        int error = errno;
        std::cerr << "error opening file '" << fname << "'" << std::endl;
        throw std::system_error(error, std::system_category());
    }

    struct stat st;
    if (fstat(fd, &st) == -1) {
        int error = errno;
        std::cerr << "error getting size of file '" << fname << "'" << std::endl;
        close(fd);
        throw std::system_error(error, std::system_category());
    }

    if (filesize) *filesize = st.st_size;
    data->length = st.st_size;

    void *mapped = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED) {
        int error = errno;
        std::cerr << "error mapping file '" << fname << "'" << std::endl;
        throw std::system_error(error, std::system_category());
    }

    data->mapping_ptr = mapped;
    return mapped;
}

static void unmap_file(FileMappingData *data) {
    if (data->mapping_ptr && data->length > 0) {
        munmap(data->mapping_ptr, data->length);
        data->mapping_ptr = nullptr;
        data->length = 0;
    }
}

#endif
