// bonsai-turbo — minimal GGUF v3 reader (mmap, zero-copy tensor access).
// Just enough to load Bonsai packs; not a general GGUF library.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace bt {

// GGUF metadata value types (spec ids).
enum class GGUFType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6,
    Bool = 7, String = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12,
};

// ggml tensor dtypes we care about (ids from the vendor fork's ggml.h).
enum class GGMLType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q8_0 = 8,
    BF16 = 30, Q1_0 = 41, Q2_0 = 42,
};

struct GGUFValue {
    GGUFType type;
    // Scalars normalized to i64/f64/bool/string; arrays kept typed + flat.
    std::variant<int64_t, double, bool, std::string,
                 std::vector<int64_t>, std::vector<double>, std::vector<std::string>> v;

    int64_t            as_int()  const { return std::get<int64_t>(v); }
    double             as_f64()  const { return std::get<double>(v); }
    bool               as_bool() const { return std::get<bool>(v); }
    const std::string& as_str()  const { return std::get<std::string>(v); }
    const std::vector<int64_t>&     as_ints() const { return std::get<std::vector<int64_t>>(v); }
    const std::vector<std::string>& as_strs() const { return std::get<std::vector<std::string>>(v); }
};

struct TensorInfo {
    std::string          name;
    std::vector<int64_t> shape;   // ne[0] is the contiguous (row) dimension
    GGMLType             type = GGMLType::F32;
    uint64_t             offset = 0;  // relative to data section
    int64_t              n_elements() const;
};

class GGUFFile {
  public:
    GGUFFile() = default;
    ~GGUFFile();
    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    // Throws std::runtime_error on parse failure.
    void open(const std::string& path);

    const std::map<std::string, GGUFValue>& metadata() const { return kv_; }
    const std::vector<TensorInfo>&          tensors()  const { return tensors_; }

    bool               has(const std::string& key) const { return kv_.count(key) != 0; }
    const GGUFValue&   value(const std::string& key) const;
    int64_t            get_int(const std::string& key, int64_t fallback) const;

    const TensorInfo*  find(const std::string& name) const;
    const uint8_t*     tensor_data(const TensorInfo& t) const { return data_ + t.offset; }
    uint64_t           tensor_nbytes(const TensorInfo& t) const;

  private:
    std::map<std::string, GGUFValue> kv_;
    std::vector<TensorInfo>          tensors_;
    const uint8_t* map_  = nullptr;  // whole file
    const uint8_t* data_ = nullptr;  // aligned tensor data section
    uint64_t       size_ = 0;
#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* map_handle_  = nullptr;
#endif
};

// Bytes per block-row helpers for the supported dtypes.
uint64_t ggml_row_nbytes(GGMLType type, int64_t row_elems);
const char* ggml_type_name(GGMLType type);

}  // namespace bt
