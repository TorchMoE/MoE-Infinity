// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

// #include <iostream>
// #include <mutex>

#include "base/logging.h"
// #include "base/noncopyable.h"

// #define PRINT_ARGS(...)                    \
//     do {                                   \
//         std::cout << #__VA_ARGS__ << ": "; \
//         print(__VA_ARGS__);                \
//     } while (0)

// // Helper function to print each argument
// inline void print() { std::cout << std::endl; }  // Base case to end
// recursion

// template <typename T, typename... Args>
// inline void print(T first, Args... args)
// {
//     std::cout << first;
//     if constexpr (sizeof...(args) > 0) {
//         std::cout << ", ";
//         print(args...);  // Recursive call
//     } else {
//         std::cout << std::endl;
//     }
// }

inline void print(base::LogStream& stream) {}

template <typename T, typename... Args>
inline void print(base::LogStream& stream, T first, Args... args) {
  stream << first;
  if constexpr (sizeof...(args) > 0) {
    stream << " ";
    print(stream, args...);  // Recursive call
  }
}

#define DLOG_TRACE(...)                                                     \
  do {                                                                      \
    if (base::Logger::logLevel() <= base::Logger::TRACE)                    \
      print(base::Logger(__FILE__, __LINE__, base::Logger::TRACE, __func__) \
                .stream(),                                                  \
            __VA_ARGS__);                                                   \
  } while (0)

#define DLOG_DEBUG(...)                                                     \
  do {                                                                      \
    if (base::Logger::logLevel() <= base::Logger::DEBUG)                    \
      print(base::Logger(__FILE__, __LINE__, base::Logger::DEBUG, __func__) \
                .stream(),                                                  \
            __VA_ARGS__);                                                   \
  } while (0)

#define DLOG_INFO(...)                                               \
  do {                                                               \
    if (base::Logger::logLevel() <= base::Logger::INFO)              \
      print(base::Logger(__FILE__, __LINE__).stream(), __VA_ARGS__); \
  } while (0)

#define DLOG_ERROR(...)                                                     \
  do {                                                                      \
    if (base::Logger::logLevel() <= base::Logger::ERROR)                    \
      print(base::Logger(__FILE__, __LINE__, base::Logger::ERROR).stream(), \
            __VA_ARGS__);                                                   \
  } while (0);

#define DLOG_WARN(...)                                                     \
  do {                                                                     \
    if (base::Logger::logLevel() <= base::Logger::WARN)                    \
      print(base::Logger(__FILE__, __LINE__, base::Logger::WARN).stream(), \
            __VA_ARGS__);                                                  \
  } while (0)

#define DLOG_FATAL(...)                                                     \
  do {                                                                      \
    if (base::Logger::logLevel() <= base::Logger::FATAL)                    \
      print(base::Logger(__FILE__, __LINE__, base::Logger::FATAL).stream(), \
            __VA_ARGS__);                                                   \
  } while (0)

// int str2level(const char* level);
// std::string level2str(int level);
// std::string formatstr();

// enum ArcherLogLevel { kFatal, kDebug, kInfo, kWarn, kError };

// extern std::once_flag kLoggerFlag;
// extern int kLogLevel;
// extern std::mutex kLogMutex;

// extern void InitLogger();

// #define DLOG_DEBUG(...)                                     \
//     do {                                                          \
//         if (kLogLevel <= kDebug) {                                \
//             std::lock_guard<std::mutex> lock(kLogMutex);          \
//             std::cout << formatstr() << level2str(kDebug) << " "; \
//             print(__VA_ARGS__);                                   \
//         }                                                         \
//     } while (0)

// #define DLOG_INFO(...)                                     \
//     do {                                                         \
//         if (kLogLevel <= kInfo) {                                \
//             std::lock_guard<std::mutex> lock(kLogMutex);         \
//             std::cout << formatstr() << level2str(kInfo) << " "; \
//             print(__VA_ARGS__);                                  \
//         }                                                        \
//     } while (0)

// #define DLOG_ERROR(...)                                     \
//     do {                                                          \
//         if (kLogLevel <= kError) {                                \
//             std::lock_guard<std::mutex> lock(kLogMutex);          \
//             std::cout << formatstr() << level2str(kError) << " "; \
//             print(__VA_ARGS__);                                   \
//         }                                                         \
//     } while (0)

// #define DLOG_WARN(...)                                     \
//     do {                                                         \
//         if (kLogLevel <= kWarn) {                                \
//             std::lock_guard<std::mutex> lock(kLogMutex);         \
//             std::cout << formatstr() << level2str(kWarn) << " "; \
//             print(__VA_ARGS__);                                  \
//         }                                                        \
//     } while (0)

// #define DLOG_FATAL(...)                                     \
//     do {                                                          \
//         if (kLogLevel <= kError) {                                \
//             std::lock_guard<std::mutex> lock(kLogMutex);          \
//             std::cout << formatstr() << level2str(kFatal) << " "; \
//             print(__VA_ARGS__);                                   \
//             throw std::runtime_error("Logged a FATAL error");     \
//         }                                                         \
//     } while (0)
