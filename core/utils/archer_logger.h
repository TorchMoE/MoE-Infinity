// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <spdlog/spdlog.h>
#include <mutex>

#include "noncopyable.h"

extern std::shared_ptr<spdlog::logger> kLogger;
extern std::once_flag kLoggerFlag;

extern void InitLogger();

#define ARCHER_LOG_INFO(...) SPDLOG_LOGGER_INFO(kLogger, __VA_ARGS__)
#define ARCHER_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(kLogger, __VA_ARGS__)
#define ARCHER_LOG_WARN(...) SPDLOG_LOGGER_WARN(kLogger, __VA_ARGS__)
#define ARCHER_LOG_DEBUG(...) kLogger->debug(__VA_ARGS__)
#define ARCHER_LOG_TRACE(...) SPDLOG_LOGGER_TRACE(kLogger, __VA_ARGS__)
#define ARCHER_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(kLogger, __VA_ARGS__)
#define ARCHER_LOG_FATAL(...)                             \
    do {                                                  \
        SPDLOG_LOGGER_CRITICAL(kLogger, __VA_ARGS__);     \
        throw std::runtime_error("Logged a FATAL error"); \
    } while (0)
