// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include "base/logging.h"

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
