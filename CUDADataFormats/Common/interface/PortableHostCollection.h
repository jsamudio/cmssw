#ifndef CUDADataFormats_Common_interface_PortableHostCollection_h
#define CUDADataFormats_Common_interface_PortableHostCollection_h

#include <cassert>
#include <cstdlib>

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "CUDADataFormats/Common/interface/PortableCollectionCommon.h"

namespace cms::cuda {

  // generic SoA-based product in host memory
  template <typename T>
  class PortableHostCollection {
  public:
    using Layout = T;
    using View = typename Layout::View;
    using ConstView = typename Layout::ConstView;
    using Buffer = cms::cuda::host::unique_ptr<std::byte[]>;

    PortableHostCollection() = default;

    PortableHostCollection(int32_t elements)
        // allocate pageable host memory
        : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout::computeDataSize(elements))},
          layout_{buffer_.get(), elements},
          view_{layout_} {
      // make_host_unique for pageable host memory uses a default alignment of 128 bytes
      assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
    }

    PortableHostCollection(int32_t elements, cudaStream_t stream)
        // allocate pinned host memory, accessible by the current device
        : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout::computeDataSize(elements), stream)},
          layout_{buffer_.get(), elements},
          view_{layout_} {
      // CUDA pinned host memory uses a default alignment of at least 128 bytes
      assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
    }

    // non-copyable
    PortableHostCollection(PortableHostCollection const&) = delete;
    PortableHostCollection& operator=(PortableHostCollection const&) = delete;

    // movable
    PortableHostCollection(PortableHostCollection&&) = default;
    PortableHostCollection& operator=(PortableHostCollection&&) = default;

    // default destructor
    ~PortableHostCollection() = default;

    // access the View
    View& view() { return view_; }
    ConstView const& view() const { return view_; }
    ConstView const& const_view() const { return view_; }

    View& operator*() { return view_; }
    ConstView const& operator*() const { return view_; }

    View* operator->() { return &view_; }
    ConstView const* operator->() const { return &view_; }

    // access the Buffer
    Buffer& buffer() { return buffer_; }
    Buffer const& buffer() const { return buffer_; }
    Buffer const& const_buffer() const { return buffer_; }

    size_t bufferSize() const { return layout_.metadata().byteSize(); }

    // part of the ROOT read streamer
    static void ROOTReadStreamer(PortableHostCollection* newObj, Layout const& layout) {
      newObj->~PortableHostCollection();
      // allocate pinned host memory using the legacy stream, that synchronises with all (blocking) streams
      new (newObj) PortableHostCollection(layout.metadata().size());
      newObj->layout_.ROOTReadStreamer(layout);
    }

  private:
    Buffer buffer_;  //!
    Layout layout_;  //
    View view_;      //!
  };

// generic SoA-based product in host memory
template <typename T0, typename... Args>
class PortableHostMultiCollection {
  template <typename T>
  static constexpr std::size_t count_t_ = portablecollection::typeCount<T, T0, Args...>;

  template <typename T>
  static constexpr std::size_t index_t_ = portablecollection::typeIndex<T, T0, Args...>;

  static constexpr std::size_t members_ = portablecollection::membersCount<T0, Args...>;

public:
  using Buffer = cms::cuda::host::unique_ptr<std::byte[]>;
  using Implementation = portablecollection::CollectionImpl<0, T0, Args...>;

  using SizesArray = std::array<int32_t, members_>;

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  using Layout = portablecollection::TypeResolver<Idx, T0, Args...>;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  using View = typename Layout<Idx>::View;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  using ConstView = typename Layout<Idx>::ConstView;

private:
  template <std::size_t Idx>
  using Leaf = portablecollection::CollectionLeaf<Idx, Layout<Idx>>;

  template <std::size_t Idx>
  Leaf<Idx>& get() {
    return static_cast<Leaf<Idx>&>(impl_);
  }

  template <std::size_t Idx>
  Leaf<Idx> const& get() const {
    return static_cast<Leaf<Idx> const&>(impl_);
  }

  template <typename T>
  portablecollection::CollectionLeaf<index_t_<T>, T>& get() {
    return static_cast<portablecollection::CollectionLeaf<index_t_<T>, T>&>(impl_);
  }

  template <typename T>
  const portablecollection::CollectionLeaf<index_t_<T>, T>& get() const {
    return static_cast<const portablecollection::CollectionLeaf<index_t_<T>, T>&>(impl_);
  }

  static int32_t computeDataSize(const std::array<int32_t, members_>& sizes) {
    int32_t ret = 0;
    portablecollection::constexpr_for<0, members_>(
        [&sizes, &ret](auto i) { ret += Layout<i>::computeDataSize(sizes[i]); });
    return ret;
  }

public:
  PortableHostMultiCollection() = default;

  PortableHostMultiCollection(int32_t elements)
      // allocate pageable host memory
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout<>::computeDataSize(elements))},
        impl_{buffer_.get(), elements} {
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  PortableHostMultiCollection(int32_t elements, cudaStream_t stream)
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout<>::computeDataSize(elements), stream)},
        impl_{buffer_.get(), elements} {
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  PortableHostMultiCollection(const std::array<int32_t, members_>& sizes)
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(computeDataSize(sizes))},
        impl_{buffer_.get(), sizes} {
    portablecollection::constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    portablecollection::constexpr_for<1, members_>(
        [&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  PortableHostMultiCollection(const std::array<int32_t, members_>& sizes, cudaStream_t stream)
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(computeDataSize(sizes), stream)},
        impl_{buffer_.get(), sizes} {
    portablecollection::constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    portablecollection::constexpr_for<1, members_>(
        [&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  // non-copyable
  PortableHostMultiCollection(PortableHostMultiCollection const&) = delete;
  PortableHostMultiCollection& operator=(PortableHostMultiCollection const&) = delete;

  // movable
  PortableHostMultiCollection(PortableHostMultiCollection&&) = default;
  PortableHostMultiCollection& operator=(PortableHostMultiCollection&&) = default;

  // default destructor
  ~PortableHostMultiCollection() = default;

  // access the View by index
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>& view() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& const_view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>& operator*() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& operator*() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>* operator->() {
    return &get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const* operator->() const {
    return &get<Idx>().view_;
  }

  // access the View by type
  template <typename T>
  typename T::View& view() {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& view() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& const_view() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::View& operator*() {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& operator*() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::View* operator->() {
    return &get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const* operator->() const {
    return &get<T>().view_;
  }

  // access the Buffer
  Buffer& buffer() { return buffer_; }
  Buffer const& buffer() const { return buffer_; }
  Buffer const& const_buffer() const { return buffer_; }

  // Extract the sizes array
  SizesArray sizes() const {
    SizesArray ret;
    portablecollection::constexpr_for<0, members_>([&](auto i) { ret[i] = get<i>().layout_.metadata().size(); });
    return ret;
  }
  // part of the ROOT read streamer
  static void ROOTReadStreamer(PortableHostMultiCollection* newObj, Implementation& onfileImpl) {
    newObj->~PortableHostMultiCollection();
    // use the global "host" object returned by cms::alpakatools::host()
    std::array<int32_t, members_> sizes;
    portablecollection::constexpr_for<0, members_>([&sizes, &onfileImpl](auto i) {
      sizes[i] = static_cast<Leaf<i> const&>(onfileImpl).layout_.metadata().size();
    });
    new (newObj) PortableHostMultiCollection(sizes);
    portablecollection::constexpr_for<0, members_>([&newObj, &onfileImpl](auto i) {
      static_cast<Leaf<i>&>(newObj->impl_).layout_.ROOTReadStreamer(static_cast<Leaf<i> const&>(onfileImpl).layout_);
      static_cast<Leaf<i>&>(onfileImpl).layout_.ROOTStreamerCleaner();
    });
  }

private:
  Buffer buffer_;  //!
  Implementation impl_;           // (serialized: this is where the layouts live)
};

// Singleton case does not need to be aliased. A special template covers it.

// This aliasing is needed to work with ROOT serialization. Bare templates make dictionary compilation fail.
template <typename T0, typename T1>
using PortableHostCollection2 = PortableHostMultiCollection<T0, T1>;

template <typename T0, typename T1, typename T2>
using PortableHostCollection3 = PortableHostMultiCollection<T0, T1, T2>;

template <typename T0, typename T1, typename T2, typename T3>
using PortableHostCollection4 = PortableHostMultiCollection<T0, T1, T2, T3>;

template <typename T0, typename T1, typename T2, typename T3, typename T4>
using PortableHostCollection5 = PortableHostMultiCollection<T0, T1, T2, T3, T4>;

}  // namespace cms::cuda

#endif  // CUDADataFormats_Common_interface_PortableHostCollection_h
