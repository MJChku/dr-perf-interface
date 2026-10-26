// SPDX-License-Identifier: MIT
// Host implementations of CUDA external PTX functions used by gxptx output.
//
// PTX `call` does not use the host C ABI. Inputs and results are addresses of
// caller-owned `.param` objects. These adapters decode that ABI before using
// host facilities.

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

extern "C" void gx_ptx_trap_op(const char* op);

namespace {

std::mutex gx_device_stdio_mu;
constexpr uint64_t gx_fake_base = 0x0000800000000000ULL;

[[noreturn]] void bad_call(const char* reason) {
    gx_ptx_trap_op(reason);
    std::abort();
}

template <typename T>
T load_unaligned(const void* p) {
    T value{};
    if (!p) bad_call("call.null-param");
    std::memcpy(&value, p, sizeof(value));
    return value;
}

const void* param_pointer(void** inputs, size_t index) {
    if (!inputs || !inputs[index]) bad_call("call.missing-param");
    uint64_t address = load_unaligned<uint64_t>(inputs[index]);
    if (address >= gx_fake_base) bad_call("call.fake-pointer");
    return reinterpret_cast<const void*>(static_cast<uintptr_t>(address));
}

size_t align_up(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

template <typename T>
T packed_arg(const unsigned char* data, size_t& offset) {
    if (!data) bad_call("vprintf.null-arg-buffer");
    constexpr size_t alignment = alignof(T) > 8 ? 8 : alignof(T);
    offset = align_up(offset, alignment);
    T value = load_unaligned<T>(data + offset);
    offset += sizeof(T);
    return value;
}

template <typename T>
void append_format(std::string& output, const std::string& spec,
                   bool width_star, int width,
                   bool precision_star, int precision, T value) {
    int needed;
    if (width_star && precision_star)
        needed = std::snprintf(nullptr, 0, spec.c_str(), width, precision, value);
    else if (width_star)
        needed = std::snprintf(nullptr, 0, spec.c_str(), width, value);
    else if (precision_star)
        needed = std::snprintf(nullptr, 0, spec.c_str(), precision, value);
    else
        needed = std::snprintf(nullptr, 0, spec.c_str(), value);
    if (needed < 0) bad_call("vprintf.host-format");
    std::vector<char> buffer(static_cast<size_t>(needed) + 1);
    if (width_star && precision_star)
        std::snprintf(buffer.data(), buffer.size(), spec.c_str(), width, precision, value);
    else if (width_star)
        std::snprintf(buffer.data(), buffer.size(), spec.c_str(), width, value);
    else if (precision_star)
        std::snprintf(buffer.data(), buffer.size(), spec.c_str(), precision, value);
    else
        std::snprintf(buffer.data(), buffer.size(), spec.c_str(), value);
    output.append(buffer.data(), static_cast<size_t>(needed));
}

bool is_flag(char c) {
    return c == '-' || c == '+' || c == ' ' || c == '#' || c == '0' || c == '\'';
}

std::string decode_vprintf(const char* format, const unsigned char* packed) {
    if (!format) bad_call("vprintf.null-format");
    std::string output;
    size_t arg_offset = 0;
    for (size_t i = 0; format[i];) {
        if (format[i] != '%') {
            output.push_back(format[i++]);
            continue;
        }

        const size_t start = i++;
        if (format[i] == '%') {
            output.push_back('%');
            ++i;
            continue;
        }
        // CUDA device printf does not document POSIX positional parameters.
        // Reject them instead of consuming the packed buffer in the wrong order.
        size_t probe = i;
        while (std::isdigit(static_cast<unsigned char>(format[probe]))) ++probe;
        if (format[probe] == '$') bad_call("vprintf.positional");

        while (is_flag(format[i])) ++i;
        bool width_star = false;
        int width = 0;
        if (format[i] == '*') {
            width_star = true;
            width = packed_arg<int32_t>(packed, arg_offset);
            ++i;
        } else {
            while (std::isdigit(static_cast<unsigned char>(format[i]))) ++i;
        }

        bool precision_star = false;
        int precision = 0;
        if (format[i] == '.') {
            ++i;
            if (format[i] == '*') {
                precision_star = true;
                precision = packed_arg<int32_t>(packed, arg_offset);
                ++i;
            } else {
                while (std::isdigit(static_cast<unsigned char>(format[i]))) ++i;
            }
        }

        enum class Length { none, hh, h, l, ll, j, z, t, L };
        Length length = Length::none;
        if (format[i] == 'h' && format[i + 1] == 'h') { length = Length::hh; i += 2; }
        else if (format[i] == 'h') { length = Length::h; ++i; }
        else if (format[i] == 'l' && format[i + 1] == 'l') { length = Length::ll; i += 2; }
        else if (format[i] == 'l') { length = Length::l; ++i; }
        else if (format[i] == 'j') { length = Length::j; ++i; }
        else if (format[i] == 'z') { length = Length::z; ++i; }
        else if (format[i] == 't') { length = Length::t; ++i; }
        else if (format[i] == 'L') { length = Length::L; ++i; }

        const char conversion = format[i];
        if (!conversion) bad_call("vprintf.incomplete-format");
        ++i;
        const std::string spec(format + start, i - start);

        switch (conversion) {
        case 'd':
        case 'i': {
            int64_t value = (length == Length::l || length == Length::ll ||
                             length == Length::j || length == Length::z ||
                             length == Length::t)
                                ? packed_arg<int64_t>(packed, arg_offset)
                                : packed_arg<int32_t>(packed, arg_offset);
            if (length == Length::l)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<long>(value));
            else if (length == Length::ll)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<long long>(value));
            else if (length == Length::j)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<intmax_t>(value));
            else if (length == Length::z)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<std::make_signed_t<size_t>>(value));
            else if (length == Length::t)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<ptrdiff_t>(value));
            else
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<int>(value));
            break;
        }
        case 'o':
        case 'u':
        case 'x':
        case 'X': {
            uint64_t value = (length == Length::l || length == Length::ll ||
                              length == Length::j || length == Length::z ||
                              length == Length::t)
                                 ? packed_arg<uint64_t>(packed, arg_offset)
                                 : packed_arg<uint32_t>(packed, arg_offset);
            if (length == Length::l)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<unsigned long>(value));
            else if (length == Length::ll)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<unsigned long long>(value));
            else if (length == Length::j)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<uintmax_t>(value));
            else if (length == Length::z)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<size_t>(value));
            else if (length == Length::t)
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<std::make_unsigned_t<ptrdiff_t>>(value));
            else
                append_format(output, spec, width_star, width, precision_star, precision,
                              static_cast<unsigned int>(value));
            break;
        }
        case 'a':
        case 'A':
        case 'e':
        case 'E':
        case 'f':
        case 'F':
        case 'g':
        case 'G':
            if (length == Length::L) bad_call("vprintf.long-double");
            append_format(output, spec, width_star, width, precision_star, precision,
                          packed_arg<double>(packed, arg_offset));
            break;
        case 'c':
            if (length == Length::l) bad_call("vprintf.wide-char");
            append_format(output, spec, width_star, width, precision_star, precision,
                          packed_arg<int32_t>(packed, arg_offset));
            break;
        case 's': {
            if (length == Length::l) bad_call("vprintf.wide-string");
            uint64_t address = packed_arg<uint64_t>(packed, arg_offset);
            if (address >= gx_fake_base) bad_call("vprintf.fake-string");
            const char* value = address
                ? reinterpret_cast<const char*>(static_cast<uintptr_t>(address))
                : "(null)";
            append_format(output, spec, width_star, width, precision_star, precision, value);
            break;
        }
        case 'p': {
            uint64_t address = packed_arg<uint64_t>(packed, arg_offset);
            void* value = reinterpret_cast<void*>(static_cast<uintptr_t>(address));
            append_format(output, spec, width_star, width, precision_star, precision, value);
            break;
        }
        case '%':
            bad_call("vprintf.decorated-percent");
        case 'n':
            bad_call("vprintf.percent-n");
        default:
            bad_call("vprintf.unsupported-format");
        }
    }
    return output;
}

}  // namespace

extern "C" void gx_ptx_vprintf(void** returns, void** inputs) {
    const char* format = static_cast<const char*>(param_pointer(inputs, 0));
    const unsigned char* packed =
        static_cast<const unsigned char*>(param_pointer(inputs, 1));
    std::string output = decode_vprintf(format, packed);
    {
        std::lock_guard<std::mutex> lock(gx_device_stdio_mu);
        if (!output.empty())
            std::fwrite(output.data(), 1, output.size(), stdout);
        std::fflush(stdout);
    }
    int32_t result = static_cast<int32_t>(output.size());
    if (returns && returns[0])
        std::memcpy(returns[0], &result, sizeof(result));
}

extern "C" void gx_ptx_assertfail(void**, void** inputs) {
    const char* assertion = static_cast<const char*>(param_pointer(inputs, 0));
    const char* file = static_cast<const char*>(param_pointer(inputs, 1));
    uint32_t line = load_unaligned<uint32_t>(inputs[2]);
    const char* function = static_cast<const char*>(param_pointer(inputs, 3));
    uint64_t char_size = load_unaligned<uint64_t>(inputs[4]);
    {
        std::lock_guard<std::mutex> lock(gx_device_stdio_mu);
        std::fprintf(stderr,
                     "GX CUDA device assertion failed: %s at %s:%u in %s "
                     "(character size %llu)\n",
                     assertion ? assertion : "(null)",
                     file ? file : "(unknown)", line,
                     function ? function : "(unknown)",
                     static_cast<unsigned long long>(char_size));
        std::fflush(stderr);
    }
    // Reaching __assertfail is the defined behavior of a failed CUDA device
    // assertion, not an untranslated PTX instruction. Terminate after the
    // CUDA-specific diagnostic without reporting a translation gap.
    std::abort();
}
