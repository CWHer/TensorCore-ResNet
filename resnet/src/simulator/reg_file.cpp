#include "reg_file.hpp"

namespace Sim {

// predicated register file
// NOTE: R7 is PT, which is always true
// clang-format off
template<> const std::unordered_map<u32, bool> PRegisterFile::SPECIAL_REG = {{7, true}};
// clang-format on

// register file
// NOTE: R255 is RZ, which is always 0
// clang-format off
template<> const std::unordered_map<u32, u32> RegisterFile::SPECIAL_REG = {{255, 0}};
// clang-format on

// special register file
// clang-format off
template<> const std::unordered_map<u32, u32> SRegisterFile::SPECIAL_REG = {};
// clang-format on

}
