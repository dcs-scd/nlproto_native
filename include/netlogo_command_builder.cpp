#include "netlogo_command_builder.hpp"
#include <sstream>
#include <iostream>

std::string build_setup_commands(
        const std::unordered_map<std::string, std::string>& params)
{
    std::ostringstream oss;
    for (const auto& [k,v] : params) {
        oss << "set " << k << " " << v << '\n';
    }
    return oss.str();
}