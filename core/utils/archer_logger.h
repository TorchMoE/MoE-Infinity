// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

// #include <spdlog/spdlog.h>
#include <iostream>
#include <mutex>

#include "noncopyable.h"

#define PRINT_ARGS(...)                    \
    do {                                   \
        std::cout << #__VA_ARGS__ << ": "; \
        print(__VA_ARGS__);                \
    } while (0)

// Helper function to print each argument
inline void print() { std::cout << std::endl; }  // Base case to end recursion

template <typename T, typename... Args>
inline void print(T first, Args... args)
{
    std::cout << first;
    if constexpr (sizeof...(args) > 0) {
        std::cout << ", ";
        print(args...);  // Recursive call
    } else {
        std::cout << std::endl;
    }
}

int str2level(const char* level);
std::string level2str(int level);
std::string formatstr();

enum ArcherLogLevel { kFatal, kDebug, kInfo, kWarn, kError };
// class ArcherLogger : public noncopyable {
// public:
//     DELETE_COPY_AND_ASSIGN(ArcherLogger);
//     STATIC_GET_INSTANCE(ArcherLogger);

//     void info(const char* fmt, ...);
//     void error(const char* fmt, ...);
//     void warn(const char* fmt, ...);
//     void debug(const char* fmt, ...);
//     void fatal(const char* fmt, ...);

// private:

//     std::string formatstr();
//     std::string level2str(int level);

// private:
//     static std::shared_ptr<spdlog::logger> kLogger;
//     static std::once_flag kLoggerFlag;
//     std::mutex log_mutex_;
//     int log_level_ = -1;
//     std::string format_str_;
// };

// extern std::shared_ptr<spdlog::logger> kLogger;
extern std::once_flag kLoggerFlag;
extern int kLogLevel;
extern std::mutex kLogMutex;

extern void InitLogger();

#define ARCHER_LOG_DEBUG(...)                                    \
    do {                                                         \
        if (kLogLevel <= kDebug) {                               \
            std::lock_guard<std::mutex> lock(kLogMutex);         \
            std::cout << formatstr() << level2str(kDebug) << " "; \
            print(__VA_ARGS__);                              \
        }                                                        \
    } while (0)

#define ARCHER_LOG_INFO(...)                                     \
    do {                                                         \
        if (kLogLevel <= kInfo) {                                \
            std::lock_guard<std::mutex> lock(kLogMutex);         \
            std::cout << formatstr() << level2str(kInfo) << " "; \
            print(__VA_ARGS__);                             \
        }                                                        \
    } while (0)

#define ARCHER_LOG_ERROR(...)                                    \
    do {                                                         \
        if (kLogLevel <= kError) {                               \
            std::lock_guard<std::mutex> lock(kLogMutex);         \
            std::cout << formatstr() << level2str(kError) << " "; \
            print(__VA_ARGS__);                         \
        }                                                        \
    } while (0)

#define ARCHER_LOG_WARN(...)                                     \
    do {                                                         \
        if (kLogLevel <= kWarn) {                                \
            std::lock_guard<std::mutex> lock(kLogMutex);         \
            std::cout << formatstr() << level2str(kWarn) << " "; \
            print(__VA_ARGS__);                              \
        }                                                        \
    } while (0)

#define ARCHER_LOG_FATAL(...)                                    \
    do {                                                         \
        if (kLogLevel <= kError) {                               \
            std::lock_guard<std::mutex> lock(kLogMutex);         \
            std::cout << formatstr() << level2str(kFatal) << " "; \
            print(__VA_ARGS__);                              \
            throw std::runtime_error("Logged a FATAL error");    \
        }                                                        \
    } while (0)

