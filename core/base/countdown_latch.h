// Use of this source code is governed by a BSD-style license
// that can be found in the License file.
//
// Author: Shuo Chen (chenshuo at chenshuo dot com)

#ifndef MUDUO_BASE_COUNTDOWNLATCH_H
#define MUDUO_BASE_COUNTDOWNLATCH_H

#include "noncopyable.h"

#include <condition_variable>
#include <mutex>

namespace base {

class CountDownLatch : noncopyable {
 public:
  explicit CountDownLatch(int count);

  void wait();

  void countDown();

  int getCount() const;

 private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  int count_;
};

}  // namespace base
#endif  // MUDUO_BASE_COUNTDOWNLATCH_H
