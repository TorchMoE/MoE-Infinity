#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>  // For std::unique_ptr

// Trie Node Definition (Template)
template <typename T>
class TrieNode {
 public:
  std::unordered_map<char, std::unique_ptr<TrieNode<T>>> children;
  std::unique_ptr<T> value;  // Store any data type
  bool is_end_of_word = false;
};

template <typename T>
class Trie {
 private:
  std::unique_ptr<TrieNode<T>> root_;

 public:
  Trie() : root_(std::make_unique<TrieNode<T>>()) {}

  // Insert a key-value pair (T can be torch::Tensor or other types)
  void Insert(const std::string& key, T value) {
    TrieNode<T>* node = root_.get();
    for (char ch : key) {
      if (node->children.find(ch) == node->children.end()) {
        node->children[ch] = std::make_unique<TrieNode<T>>();
      }
      node = node->children[ch].get();
    }
    node->is_end_of_word = true;
    node->value = std::make_unique<T>(std::move(value));  // Move value
  }

  // Retrieve all keys with a given prefix
  void SearchPrefix(TrieNode<T>* node, std::vector<std::string>& results,
                    std::string current) {
    if (node->is_end_of_word) {
      results.push_back(current);
    }
    for (auto& [ch, child] : node->children) {
      SearchPrefix(child.get(), results, current + ch);
    }
  }

  // Get all keys with a given prefix
  std::vector<std::string> GetKeysWithPrefix(const std::string& prefix) {
    TrieNode<T>* node = root_.get();
    std::vector<std::string> results;

    for (char ch : prefix) {
      if (node->children.find(ch) == node->children.end()) {
        return results;  // No match
      }
      node = node->children[ch].get();
    }
    SearchPrefix(node, results, prefix);
    return results;
  }

  // Get a value by exact key
  T Get(const std::string& key) {
    TrieNode<T>* node = root_.get();
    for (char ch : key) {
      if (node->children.find(ch) == node->children.end()) {
        throw std::runtime_error("Key not found: " + key);
      }
      node = node->children[ch].get();
    }
    if (!node->is_end_of_word) {
      throw std::runtime_error("Key not found: " + key);
    }
    return *(node->value);  // Return the stored value
  }
};
