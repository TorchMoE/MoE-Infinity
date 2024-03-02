// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_logger.h"

// #include <spdlog/sinks/stdout_color_sinks.h>
// #include <spdlog/fmt/bundled/ostream.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

// std::shared_ptr<spdlog::logger> kLogger = nullptr;
std::once_flag kLoggerFlag;
int kLogLevel = -1;
std::mutex kLogMutex;

// static const char* kArcherLoggerName = "archer_logger";

int str2level(const char* level)
{
    if (strcmp(level, "info") == 0) {
        return kInfo;
    } else if (strcmp(level, "error") == 0) {
        return kError;
    } else if (strcmp(level, "warn") == 0) {
        return kWarn;
    } else if (strcmp(level, "debug") == 0) {
        return kDebug;
    } else if (strcmp(level, "fatal") == 0) {
        return kFatal;
    } else {
        return -1;
    }
}

std::string level2str(int level)
{
    switch (level) {
        case kInfo:
            return "INFO";
        case kError:
            return "ERROR";
        case kWarn:
            return "WARN";
        case kDebug:
            return "DEBUG";
        case kFatal:
            return "FATAL";
        default:
            return "UNKNOWN";
    }
}

std::string formatstr()
{
    // std::string format = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] ";
    
    // get actual values in the format
    auto time = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(time);
    auto tm = *std::localtime(&timer);
    
    auto year = tm.tm_year + 1900;
    auto month = tm.tm_mon + 1;
    auto day = tm.tm_mday;
    auto hour = tm.tm_hour;
    auto min = tm.tm_min;
    auto sec = tm.tm_sec;
    auto msec = ms.count();

    char buf[128];
    sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d.%03ld", year, month, day, hour, min, sec, msec);
    return std::string(buf) + " ";
}

// void ArcherLogger::info(const char* fmt, ...)
// {
//     std::lock_guard<std::mutex> lock(log_mutex_);
//     if (log_level_ < 0) {
//         const char* level = getenv("LOG_LEVEL");
//         if (level) {
//             log_level_ = str2level(level);
//         } else {
//             log_level_ = kInfo;
//         }
//     }

//     auto format_str = formatstr();

//     if (log_level_ <= kInfo) {
//         std::cout << format_str;

//     }
    
// }

void InitLogger()
{
    std::call_once(kLoggerFlag, []() {
        // kLogger = spdlog::get(kArcherLoggerName);
        // kLogger = spdlog::stdout_color_mt(kArcherLoggerName);
        // printf("SPDLOG_LEVEL : %s\n", getenv("SPDLOG_LEVEL"));
        // if (getenv("SPDLOG_LEVEL")) {
        //     kLogger->set_level(spdlog::level::from_str(getenv("SPDLOG_LEVEL")));
        // } else {
        //     kLogger->set_level(spdlog::level::info);
        // }
        // kLogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        // kLogger->debug("create logger for MoE-Infinity");

        printf("SPDLOG_LEVEL : %s\n", getenv("SPDLOG_LEVEL"));
        if (getenv("SPDLOG_LEVEL")) {
            kLogLevel = str2level(getenv("SPDLOG_LEVEL"));
        } else {
            kLogLevel = kInfo;
        }
    });
}
