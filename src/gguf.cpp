#include "gguf.h"

#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace bt {

namespace {

constexpr uint32_t kMagic = 0x46554747;  // "GGUF"

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;

    void need(uint64_t n) const {
        if ((uint64_t)(end - p) < n) throw std::runtime_error("gguf: truncated file");
    }
    template <typename T>
    T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string read_str() {
        const uint64_t n = read<uint64_t>();
        need(n);
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
};

GGUFValue read_value(Cursor& c, GGUFType type) {
    GGUFValue out{type, int64_t{0}};
    switch (type) {
        case GGUFType::U8:     out.v = (int64_t)c.read<uint8_t>();  break;
        case GGUFType::I8:     out.v = (int64_t)c.read<int8_t>();   break;
        case GGUFType::U16:    out.v = (int64_t)c.read<uint16_t>(); break;
        case GGUFType::I16:    out.v = (int64_t)c.read<int16_t>();  break;
        case GGUFType::U32:    out.v = (int64_t)c.read<uint32_t>(); break;
        case GGUFType::I32:    out.v = (int64_t)c.read<int32_t>();  break;
        case GGUFType::U64:    out.v = (int64_t)c.read<uint64_t>(); break;
        case GGUFType::I64:    out.v = c.read<int64_t>();           break;
        case GGUFType::F32:    out.v = (double)c.read<float>();     break;
        case GGUFType::F64:    out.v = c.read<double>();            break;
        case GGUFType::Bool:   out.v = c.read<uint8_t>() != 0;      break;
        case GGUFType::String: out.v = c.read_str();                break;
        case GGUFType::Array: {
            const auto elem = (GGUFType)c.read<uint32_t>();
            const uint64_t n = c.read<uint64_t>();
            if (elem == GGUFType::String) {
                std::vector<std::string> a;
                a.reserve(n);
                for (uint64_t i = 0; i < n; ++i) a.push_back(c.read_str());
                out.v = std::move(a);
            } else if (elem == GGUFType::F32 || elem == GGUFType::F64) {
                std::vector<double> a;
                a.reserve(n);
                for (uint64_t i = 0; i < n; ++i) {
                    a.push_back(std::get<double>(read_value(c, elem).v));
                }
                out.v = std::move(a);
            } else if (elem == GGUFType::Array) {
                throw std::runtime_error("gguf: nested arrays unsupported");
            } else {
                std::vector<int64_t> a;
                a.reserve(n);
                for (uint64_t i = 0; i < n; ++i) {
                    GGUFValue e = read_value(c, elem);
                    a.push_back(std::holds_alternative<bool>(e.v) ? (int64_t)e.as_bool()
                                                                  : e.as_int());
                }
                out.v = std::move(a);
            }
            break;
        }
        default:
            throw std::runtime_error("gguf: unknown value type " +
                                     std::to_string((uint32_t)type));
    }
    return out;
}

}  // namespace

int64_t TensorInfo::n_elements() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

uint64_t ggml_row_nbytes(GGMLType type, int64_t row_elems) {
    switch (type) {
        case GGMLType::F32:  return (uint64_t)row_elems * 4;
        case GGMLType::F16:
        case GGMLType::BF16: return (uint64_t)row_elems * 2;
        case GGMLType::Q4_0: return (uint64_t)(row_elems / 32) * 18;
        case GGMLType::Q4_1: return (uint64_t)(row_elems / 32) * 20;
        case GGMLType::Q8_0: return (uint64_t)(row_elems / 32) * 34;
        case GGMLType::Q1_0: return (uint64_t)(row_elems / 128) * 18;
        case GGMLType::Q2_0: return (uint64_t)(row_elems / 128) * 34;
    }
    throw std::runtime_error("gguf: row size for unsupported dtype");
}

const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return "f32";
        case GGMLType::F16:  return "f16";
        case GGMLType::BF16: return "bf16";
        case GGMLType::Q4_0: return "q4_0";
        case GGMLType::Q4_1: return "q4_1";
        case GGMLType::Q8_0: return "q8_0";
        case GGMLType::Q1_0: return "q1_0";
        case GGMLType::Q2_0: return "q2_0";
    }
    return "unknown";
}

GGUFFile::~GGUFFile() {
#ifdef _WIN32
    if (map_) UnmapViewOfFile((LPCVOID)map_);
    if (map_handle_) CloseHandle((HANDLE)map_handle_);
    if (file_handle_) CloseHandle((HANDLE)file_handle_);
#else
    if (map_) munmap((void*)map_, size_);
#endif
}

void GGUFFile::open(const std::string& path) {
#ifdef _WIN32
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("gguf: cannot open " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    size_ = (uint64_t)sz.QuadPart;
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) throw std::runtime_error("gguf: CreateFileMapping failed");
    map_ = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!map_) throw std::runtime_error("gguf: MapViewOfFile failed");
    file_handle_ = f;
    map_handle_ = m;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("gguf: fstat failed");
    }
    size_ = (uint64_t)st.st_size;
    void* m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("gguf: mmap failed");
    map_ = (const uint8_t*)m;
#endif

    Cursor c{map_, map_ + size_};
    if (c.read<uint32_t>() != kMagic) throw std::runtime_error("gguf: bad magic");
    const uint32_t version = c.read<uint32_t>();
    if (version != 3) throw std::runtime_error("gguf: unsupported version " + std::to_string(version));

    const uint64_t n_tensors = c.read<uint64_t>();
    const uint64_t n_kv      = c.read<uint64_t>();

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.read_str();
        const auto type = (GGUFType)c.read<uint32_t>();
        kv_.emplace(std::move(key), read_value(c, type));
    }

    tensors_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo t;
        t.name = c.read_str();
        const uint32_t ndims = c.read<uint32_t>();
        if (ndims > 4) throw std::runtime_error("gguf: bad ndims for " + t.name);
        for (uint32_t d = 0; d < ndims; ++d) t.shape.push_back((int64_t)c.read<uint64_t>());
        t.type   = (GGMLType)c.read<uint32_t>();
        t.offset = c.read<uint64_t>();
        tensors_.push_back(std::move(t));
    }

    const uint64_t align = (uint64_t)get_int("general.alignment", 32);
    const uint64_t header_end = (uint64_t)(c.p - map_);
    data_ = map_ + ((header_end + align - 1) / align) * align;
}

const GGUFValue& GGUFFile::value(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("gguf: missing key " + key);
    return it->second;
}

int64_t GGUFFile::get_int(const std::string& key, int64_t fallback) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? fallback : it->second.as_int();
}

const TensorInfo* GGUFFile::find(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

uint64_t GGUFFile::tensor_nbytes(const TensorInfo& t) const {
    int64_t rows = 1;
    for (size_t i = 1; i < t.shape.size(); ++i) rows *= t.shape[i];
    return ggml_row_nbytes(t.type, t.shape.empty() ? 0 : t.shape[0]) * (uint64_t)rows;
}

}  // namespace bt
