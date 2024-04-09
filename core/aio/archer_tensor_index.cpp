// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_tensor_index.h"
#include <stdio.h>

using namespace std;

void write_options(std::ostream& os, const torch::TensorOptions& obj)
{
    bool pinned_memory = obj.pinned_memory();
    bool requires_grad = obj.requires_grad();
    std::int8_t dtype = static_cast<std::int8_t>(obj.dtype().toScalarType());
    std::int8_t device_index = static_cast<std::int8_t>(obj.device().index());
    std::int8_t device_type = static_cast<std::int8_t>(obj.device().type());
    std::int8_t layout = static_cast<std::int8_t>(obj.layout());

    os.write(reinterpret_cast<char*>(&pinned_memory), sizeof(pinned_memory));
    os.write(reinterpret_cast<char*>(&requires_grad), sizeof(requires_grad));
    os.write(reinterpret_cast<char*>(&dtype), sizeof(dtype));
    os.write(reinterpret_cast<char*>(&device_index), sizeof(device_index));
    os.write(reinterpret_cast<char*>(&device_type), sizeof(device_type));
    os.write(reinterpret_cast<char*>(&layout), sizeof(layout));
}

void read_options(std::istream& is, torch::TensorOptions& obj)
{
    bool pinned_memory = obj.pinned_memory();
    bool requires_grad = obj.requires_grad();
    std::int8_t dtype = static_cast<std::int8_t>(obj.dtype().toScalarType());
    std::int8_t device_index = static_cast<std::int8_t>(obj.device().index());
    std::int8_t device_type = static_cast<std::int8_t>(obj.device().type());
    std::int8_t layout = static_cast<std::int8_t>(obj.layout());

    is.read(reinterpret_cast<char*>(&pinned_memory), sizeof(pinned_memory));
    is.read(reinterpret_cast<char*>(&requires_grad), sizeof(requires_grad));
    is.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
    is.read(reinterpret_cast<char*>(&device_index), sizeof(device_index));
    is.read(reinterpret_cast<char*>(&device_type), sizeof(device_type));
    is.read(reinterpret_cast<char*>(&layout), sizeof(layout));

    obj = obj.dtype(static_cast<c10::ScalarType>(dtype))
              .device(torch::Device(static_cast<torch::DeviceType>(device_type),
                                    static_cast<torch::DeviceIndex>(device_index)))
              .layout(static_cast<c10::Layout>(layout))
              .requires_grad(requires_grad)
              .pinned_memory(pinned_memory);
}

std::ostream& operator<<(std::ostream& os, const TensorStorageMeta& obj)
{
    os.write(reinterpret_cast<const char*>(&obj.file_id), sizeof(obj.file_id));
    os.write(reinterpret_cast<const char*>(&obj.offset), sizeof(obj.offset));
    os.write(reinterpret_cast<const char*>(&obj.size), sizeof(obj.size));

    // Write shape
    std::int64_t shape_size = obj.shape.size();
    os.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    for (const auto& dim : obj.shape) {
        os.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }

    // Write options
    write_options(os, obj.options);

    return os;
}

std::istream& operator>>(std::istream& is, TensorStorageMeta& obj)
{
    is.read(reinterpret_cast<char*>(&obj.file_id), sizeof(obj.file_id));
    is.read(reinterpret_cast<char*>(&obj.offset), sizeof(obj.offset));
    is.read(reinterpret_cast<char*>(&obj.size), sizeof(obj.size));

    // Read shape
    std::int64_t shape_size;
    is.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    obj.shape.resize(shape_size);
    for (auto& dim : obj.shape) { is.read(reinterpret_cast<char*>(&dim), sizeof(dim)); }

    // Read options
    read_options(is, obj.options);

    return is;
}

std::string TensorStorageMeta::DebugString() const
{
    std::stringstream ss;
    ss << "file_id: " << file_id << ", offset: " << offset << ", size: " << size << ", shape: [";
    for (auto& dim : shape) { ss << dim << ", "; }
    ss << "], id: " << id;
    return ss.str();
}

ArcherTensorIndex* kTensorIndex = ArcherTensorIndex::GetInstance();

void ArcherTensorIndex::Serialize(const char* path)
{
    std::uint32_t size = this->size();
    std::ofstream ofs(path, std::ios::binary | std::ios::out | std::ios::trunc);
    ofs.write(reinterpret_cast<char*>(&size), sizeof(size));
    for (auto& item : *this) {
        ofs.write(reinterpret_cast<const char*>(&item.first), sizeof(item.first));
        ofs << item.second;
    }
}

void ArcherTensorIndex::Deserialize(const char* path)
{
    this->clear();

    std::ifstream ifs(path, std::ios::binary | std::ios::out);

    std::uint32_t size;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));

    // Then read each key-value pair
    for (std::uint32_t i = 0; i < size; ++i) {
        // Read the key
        std::uint32_t key;
        ifs.read(reinterpret_cast<char*>(&key), sizeof(key));

        // Read the value
        TensorStorageMeta value;
        ifs >> value;

        // Insert into the map
        this->insert({key, value});
    }
}
