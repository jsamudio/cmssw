#ifndef DataFormats_Portable_interface_PortableDeviceCollection_h
#define DataFormats_Portable_interface_PortableDeviceCollection_h

#include <cassert>
#include <optional>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// generic SoA-based product in device memory
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
class PortableDeviceCollection {
  static_assert(not std::is_same_v<TDev, alpaka_common::DevHost>,
                "Use PortableHostCollection<T> instead of PortableDeviceCollection<T, DevHost>");

public:
  using Layout = T;
  using View = typename Layout::View;
  using ConstView = typename Layout::ConstView;
  using Buffer = cms::alpakatools::device_buffer<TDev, std::byte[]>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, std::byte[]>;
  using Implementation = CollectionImpl<0, T0, T1, T2, T3, T4>;
  using TypeResolver = CollectionTypeResolver<T0, T1, T2, T3, T4>;
  using IdxResolver = CollectionIdxResolver<T0, T1, T2, T3, T4>;
  using SizesArray = std::array<int32_t, membersCount>;

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  using Layout = typename TypeResolver::template Resolver<Idx>::type;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  using View = typename Layout<Idx>::View;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  using ConstView = typename Layout<Idx>::ConstView;

private:
  template <std::size_t Idx>
  CollectionLeaf<Idx, typename TypeResolver::template Resolver<Idx>::type>& get() {
    return dynamic_cast<CollectionLeaf<Idx, typename TypeResolver::template Resolver<Idx>::type>&>(impl_);
  }

  template <std::size_t Idx>
  const CollectionLeaf<Idx, typename TypeResolver::template Resolver<Idx>::type>& get() const {
    return dynamic_cast<const CollectionLeaf<Idx, typename TypeResolver::template Resolver<Idx>::type>&>(impl_);
  }

  template <typename T>
  CollectionLeaf<IdxResolver::template Resolver<T>::Idx, T>& get() {
    return dynamic_cast<CollectionLeaf<IdxResolver::template Resolver<T>::Idx, T>&>(impl_);
  }

  template <typename T>
  const CollectionLeaf<IdxResolver::template Resolver<T>::Idx, T>& get() const {
    return dynamic_cast<const CollectionLeaf<IdxResolver::template Resolver<T>::Idx, T>&>(impl_);
  }

public:
  PortableDeviceCollection() = default;

  PortableDeviceCollection(int32_t elements, TDev const& device)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, Layout::computeDataSize(elements))},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  PortableDeviceCollection(int32_t elements, TQueue const& queue)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(membersCount == 1);
  }

  static int32_t computeDataSize(const SizesArray& sizes) {
    int32_t ret = 0;
    constexpr_for<0, membersCount>(
        [&sizes, &ret](auto i) { ret += TypeResolver::template Resolver<i>::type::computeDataSize(sizes[i]); });
    return ret;
  }

  PortableDeviceCollection(const SizesArray& sizes, TDev const& device)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, computeDataSize(sizes))},
        impl_{buffer_->data(), sizes} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    constexpr_for<0, membersCount>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, membersCount>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  template <typename TQueue, typename = std::enable_if_t<cms::alpakatools::is_queue_v<TQueue>>>
  PortableDeviceCollection(const SizesArray& sizes, TQueue const& queue)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, computeDataSize(sizes))},
        impl_{buffer_->data(), sizes} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    constexpr_for<0, membersCount>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, membersCount>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  // non-copyable
  PortableDeviceCollection(PortableDeviceCollection const&) = delete;
  PortableDeviceCollection& operator=(PortableDeviceCollection const&) = delete;

  // movable
  PortableDeviceCollection(PortableDeviceCollection&&) = default;
  PortableDeviceCollection& operator=(PortableDeviceCollection&&) = default;

  // default destructor
  ~PortableDeviceCollection() = default;

  // access the View by index
  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  View<Idx>& view() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  ConstView<Idx> const& view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  ConstView<Idx> const& const_view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  View<Idx>& operator*() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  ConstView<Idx> const& operator*() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  View<Idx>* operator->() {
    return &get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  ConstView<Idx> const* operator->() const {
    return &get<Idx>().view_;
  }

  View& operator*() { return view_; }
  ConstView const& operator*() const { return view_; }

  View* operator->() { return &view_; }
  ConstView const* operator->() const { return &view_; }

  // access the Buffer
  Buffer buffer() { return *buffer_; }
  ConstBuffer buffer() const { return *buffer_; }
  ConstBuffer const_buffer() const { return *buffer_; }

  // Extract the sizes array
  SizesArray sizes() const {
    SizesArray ret;
    constexpr_for<0, membersCount>([&](auto i) { ret[i] = get<i>().layout_.metadata().size(); });
    return ret;
  }

private:
  std::optional<Buffer> buffer_;  //!
  Layout layout_;                 //
  View view_;                     //!
};

#endif  // DataFormats_Portable_interface_PortableDeviceCollection_h
