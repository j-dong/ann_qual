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

struct FileMappingData {
    HANDLE hFileMapping;
    void *mapping_ptr;
};

static void *map_file(const char *fname, FileMappingData *data, size_t *filesize);
static void unmap_file(FileMappingData *data);
#else
struct FileMappingData {
};

static void *map_file(const char *fname, FileMappingData *data, size_t *filesize);
static void unmap_file(FileMappingData *data);
#endif

RawVectorData data_base;
RawVectorData data_query;
RawVectorData data_learn;

static FileMappingData mapping_data[3];

void load_vector(const char *fname, RawVectorData *data, FileMappingData *map) {
    data->filestart = map_file(fname, map, &data->filesize);
    data->vec = (float *) data->filestart;
    data->dim = *(int *) data->filestart;
    data->length = data->filesize / 4 / (1 + data->dim);
}

void load_files() {
    load_vector("G:\\vectors\\siftsmall\\siftsmall_base.fvecs", &data_base, &mapping_data[0]);
    try {
        load_vector("G:\\vectors\\siftsmall\\siftsmall_query.fvecs", &data_query, &mapping_data[1]);
        try {
            load_vector("G:\\vectors\\siftsmall\\siftsmall_learn.fvecs", &data_learn, &mapping_data[2]);
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
#endif
