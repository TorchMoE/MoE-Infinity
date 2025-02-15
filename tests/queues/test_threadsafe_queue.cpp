#include "utils/threadsafe_queue.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>

TEST(ThreadSafeQueueTest, SingleThreadedPushPop) {
  ThreadSafeQueue<int> queue;
  int value;

  int a = 1;
  queue.Push(a);
  ASSERT_TRUE(queue.Pop(value));
  ASSERT_EQ(value, 1);
}

TEST(ThreadSafeQueueTest, SequentialPushParallelPop) {
  ThreadSafeQueue<int> queue;
  int value;

  // Sequential push
  for (int i = 0; i < 10; ++i) {
    queue.Push(i);
  }

  // Parallel pop
  std::vector<std::thread> threads;
  std::vector<int> results(10);
  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&queue, &results, i]() {
      int val;
      while (!queue.Pop(val)) {
        // Busy wait
      }
      results[i] = val;
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  // Verify results
  std::sort(results.begin(), results.end());
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(results[i], i);
  }
}

TEST(ThreadSafeQueueTest, ParallelPushSequentialPop) {
  ThreadSafeQueue<int> queue;
  int value;

  // Parallel push
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&queue, i]() {
      int val = i;
      queue.Push(val);
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  // Sequential pop
  std::vector<int> results(10);
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(queue.Pop(value));
    results[i] = value;
  }
  ASSERT_FALSE(queue.TryPop(value));  // Queue should be empty

  // Verify results
  std::sort(results.begin(), results.end());
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(results[i], i);
  }
}

TEST(ThreadSafeQueueTest, ParallelPushParallelPop) {
  ThreadSafeQueue<int> queue;
  int value;

  // Parallel push
  std::vector<std::thread> push_threads;
  for (int i = 0; i < 10; ++i) {
    push_threads.emplace_back([&queue, i]() {
      int val = i;
      queue.Push(val);
    });
  }

  // Parallel pop
  std::vector<std::thread> pop_threads;
  std::vector<int> results(10);
  for (int i = 0; i < 10; ++i) {
    pop_threads.emplace_back([&queue, &results, i]() {
      int val;
      while (!queue.Pop(val)) {
        // Busy wait
      }
      results[i] = val;
    });
  }

  for (auto& t : push_threads) {
    t.join();
  }
  for (auto& t : pop_threads) {
    t.join();
  }

  // Verify results
  std::sort(results.begin(), results.end());
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(results[i], i);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
