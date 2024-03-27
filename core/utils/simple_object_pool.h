#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <stack>

/** Pool for objects that cannot be used from different threads simultaneously.
 * Allows to create an object for each thread.
 * Pool has unbounded size and objects are not destroyed before destruction of pool.
 *
 * Use it in cases when thread local storage is not appropriate
 *  (when maximum number of simultaneously used objects is less
 *   than number of running/sleeping threads, that has ever used object,
 *   and creation/destruction of objects is expensive).
 */
template <typename T>
class SimpleObjectPool {
protected:
    /// Hold all available objects in stack.
    std::mutex mutex;
    std::stack<std::unique_ptr<T>> stack;

    /// Specialized deleter for std::unique_ptr.
    /// Returns underlying pointer back to stack thus reclaiming its ownership.
    struct Deleter {
        SimpleObjectPool<T>* parent;

        Deleter(SimpleObjectPool<T>* parent_ = nullptr) : parent{parent_} {}  /// NOLINT

        void operator()(T* owning_ptr) const
        {
            std::lock_guard<std::mutex> lock{parent->mutex};
            parent->stack.emplace(owning_ptr);
        }
    };

public:
    using Pointer = std::unique_ptr<T, Deleter>;

    /// Extracts and returns a pointer from the stack if it's not empty,
    ///  creates a new one by calling provided f() otherwise.
    template <typename Factory>
    Pointer get(Factory&& f)
    {
        std::unique_lock<std::mutex> lock(mutex);

        if (stack.empty()) {
            lock.unlock();
            return {f(), this};
        }

        auto object = stack.top().release();
        stack.pop();

        return {object, this};
    }

    /// Return a vector of pointers from the stack if it's not empty,
    ///  creates a new one by calling provided f() otherwise.
    template <typename Factory>
    std::vector<Pointer> getMany(size_t count, Factory&& f)
    {
        std::unique_lock<std::mutex> lock(mutex);

        std::vector<Pointer> result;
        result.reserve(count);

        while (count > 0) {
            if (stack.empty()) {
                lock.unlock();
                while (count > 0) {
                    result.emplace_back(f(), this);
                    --count;
                }
                return result;
            }

            auto object = stack.top().release();
            stack.pop();
            result.emplace_back(object, this);
            --count;
        }

        return result;
    }

    /// Like get(), but creates object using default constructor.
    Pointer getDefault()
    {
        return get([] { return new T; });
    }

    /// Like getMany(), but creates objects using default constructor.
    std::vector<Pointer> getDefaultMany(size_t count)
    {
        return getMany(count, [] { return new T; });
    }
};
