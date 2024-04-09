// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <mutex>

class noncopyable {
public:
    noncopyable(const noncopyable&) = delete;
    noncopyable& operator=(const noncopyable&) = delete;

    noncopyable(noncopyable&&) = delete;
    noncopyable& operator=(noncopyable&&) = delete;

protected:
    noncopyable() = default;
    ~noncopyable() = default;
};

#define DELETE_COPY_AND_ASSIGN(classname)            \
    classname(const classname&) = delete;            \
    classname& operator=(const classname&) = delete; \
    classname(classname&&) = delete;                 \
    classname& operator=(classname&&) = delete;

#define STATIC_GET_INSTANCE(classname)                              \
    static classname* GetInstance()                                 \
    {                                                               \
        static std::once_flag flag;                                 \
        static classname* instance = nullptr;                       \
        std::call_once(flag, []() { instance = new classname(); }); \
        return instance;                                            \
    }

template <typename T>
struct DoNothingDeleter {
    void operator()(T* ptr) const {}
};
