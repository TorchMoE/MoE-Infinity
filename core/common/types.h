// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <cstdint>

typedef std::uint32_t TensorID;
typedef std::size_t HashID;
typedef std::size_t NodeID;
typedef std::uint64_t GraphID;
typedef std::uint64_t RequestID;

#define KB 1024
#define MB (KB * KB)
#define GB (KB * KB * KB)

#define DELETE_COPY_AND_ASSIGN(classname)          \
  classname(const classname&) = delete;            \
  classname& operator=(const classname&) = delete; \
  classname(classname&&) = delete;                 \
  classname& operator=(classname&&) = delete;

#define STATIC_GET_INSTANCE(classname)                          \
  static classname* GetInstance() {                             \
    static std::once_flag flag;                                 \
    static classname* instance = nullptr;                       \
    std::call_once(flag, []() { instance = new classname(); }); \
    return instance;                                            \
  }

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};
