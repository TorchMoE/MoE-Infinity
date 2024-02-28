// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_logger.h"

#include <spdlog/sinks/stdout_color_sinks.h>
// #include <spdlog/fmt/bundled/ostream.h>
#include <stdio.h>

std::shared_ptr<spdlog::logger> kLogger = nullptr;
std::once_flag kLoggerFlag;

static const char* kArcherLoggerName = "archer_logger";

void InitLogger()
{
    std::call_once(kLoggerFlag, []() {
        kLogger = spdlog::get(kArcherLoggerName);
        kLogger = spdlog::stdout_color_mt(kArcherLoggerName);
        printf("SPDLOG_LEVEL : %s\n", getenv("SPDLOG_LEVEL"));
        if (getenv("SPDLOG_LEVEL")) {
            kLogger->set_level(spdlog::level::from_str(getenv("SPDLOG_LEVEL")));
        } else {
            kLogger->set_level(spdlog::level::info);
        }
        kLogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        kLogger->debug("create logger for MoE-Infinity");
    });
}
