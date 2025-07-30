#pragma once
#include <string>
#include <unordered_map>

std::string build_setup_commands(
        const std::unordered_map<std::string, std::string>& params);