// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <string>
#include <unordered_map>

enum StatusType : std::uint32_t {
    kOK = 0,
    kUnknown,
    kErrCuda,
};

static const std::unordered_map<StatusType, std::string> kStatusStr = {{kOK, ""},
                                                                       {kUnknown, "unknown: "},
                                                                       {kErrCuda, "cuda error: "}};

class Status {
public:
    Status() : status_(kOK), err_() {}
    bool OK() const { return status_ == kOK; }
    const uint32_t status() const { return status_; }
    const std::string& err() const { return err_; }
    void SetError(StatusType status, const std::string& msg)
    {
        status_ = status;
        if (kStatusStr.find(status) == kStatusStr.end()) status_ = kUnknown;
        err_ = kStatusStr.at(status_) + msg;
    }

private:
    StatusType status_;
    std::string err_;
};
