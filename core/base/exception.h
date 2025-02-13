// Use of this source code is governed by a BSD-style license
// that can be found in the License file.
//
// Author: Shuo Chen (chenshuo at chenshuo dot com)

#ifndef MUDUO_BASE_EXCEPTION_H
#define MUDUO_BASE_EXCEPTION_H

#include <exception>
#include <string>
#include "types.h"

namespace base {

class Exception : public std::exception {
 public:
  explicit Exception(const char* what);
  explicit Exception(const std::string& what);
  virtual ~Exception() throw();
  virtual const char* what() const throw();
  const char* stackTrace() const throw();

 private:
  void fillStackTrace();

  std::string message_;
  std::string stack_;
};

}  // namespace base

#endif  // MUDUO_BASE_EXCEPTION_H
