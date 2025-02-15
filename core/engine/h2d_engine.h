#pragma once

#include "base/noncopyable.h"
#include "base/thread_pool.h"

class H2DEngine : public base::noncopyable {
 public:
  H2DEngine(int id, int num_threads = 1) {
    std::string thread_name = std::string("H2DEngine-") + std::to_string(id);
    // use move constructor to avoid copy
    thread_pool_ = std::make_unique<base::ThreadPool>(thread_name);
    thread_pool_->start(num_threads);
  };

 private:
  std::unique_ptr<base::ThreadPool> thread_pool_;
};
