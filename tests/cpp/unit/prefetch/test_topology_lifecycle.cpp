#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "model/model_topology.h"
#include "prefetch/archer_prefetch_handle.h"

namespace {

std::string MakePrefix(const char* name) {
  std::string prefix = std::string("/tmp/moe_topo_lifecycle_") + name + "/";
  std::filesystem::create_directories(prefix);
  return prefix;
}

TEST(TopologyLifecycle, SecondConstructionThrowsInsteadOfCorrupting) {
  ArcherPrefetchHandle first(MakePrefix("a"), 0.5);
  ASSERT_NE(kTopologyHandle, nullptr);
  EXPECT_THROW(
      { ArcherPrefetchHandle second(MakePrefix("a2"), 0.5); },
      std::runtime_error);
  EXPECT_NE(kTopologyHandle, nullptr);
}

TEST(TopologyLifecycle, ConstructionAfterDestroyIsAllowed) {
  { ArcherPrefetchHandle first(MakePrefix("b"), 0.5); }
  EXPECT_EQ(kTopologyHandle, nullptr);
  EXPECT_NO_THROW({ ArcherPrefetchHandle again(MakePrefix("b2"), 0.5); });
}

}  // namespace
