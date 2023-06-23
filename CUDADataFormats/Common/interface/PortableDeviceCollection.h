#ifndef CUDADataFormats_Common_interface_PortableDeviceCollection_h
#define CUDADataFormats_Common_interface_PortableDeviceCollection_h

#include <cassert>
#include <cstdlib>
#include <optional>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "CUDADataFormats/Common/interface/PortableCollectionCommon.h"

namespace cms::cuda {

  // generic SoA-based product in device memory
  template <typename T>
  class PortableDeviceCollection {
  public:
    using Layout = T;
    using View = typename Layout::View;
    using ConstView = typename Layout::ConstView;
    using Buffer = cms::cuda::device::unique_ptr<std::byte[]>;

    PortableDeviceCollection() = default;

    PortableDeviceCollection(int32_t elements, cudaStream_t stream)
        : buffer_{cms::cuda::make_device_unique<std::byte[]>(Layout::computeDataSize(elements), stream)},
          layout_{buffer_.get(), elements},
          view_{layout_} {
      // CUDA device memory uses a default alignment of at least 128 bytes
      assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
    }

    // non-copyable
    PortableDeviceCollection(PortableDeviceCollection const&) = delete;
    PortableDeviceCollection& operator=(PortableDeviceCollection const&) = delete;

    // movable
    PortableDeviceCollection(PortableDeviceCollection&&) = default;
    PortableDeviceCollection& operator=(PortableDeviceCollection&&) = default;

    // default destructor
    ~PortableDeviceCollection() = default;

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

  private:
    Buffer buffer_;  //!
    Layout layout_;  //
    View view_;      //!
  };

  //generic SoA-Based product in device memory
  template <typename T0, typename... Args>
  class PortableDeviceMultiCollection {
      template <typename T>
      static constexpr std::size_t count_t_ = portablecollection::typeCount<T, T0, Args...>;

      template <typename T>
      static constexpr std::size_t index_t_ = portablecollection::typeIndex<T, T0, Args...>;

      static constexpr std::size_t members_ = sizeof...(Args) + 1;

  public:
      using Buffer = cms::cuda::device::unique_ptr<std::byte[]>;
      using Implementation = portablecollection::CollectionImpl<0, T0, Args...>;

      using SizesArray = std::array<int32_t, members_>;

      template <std::size_t Idx = 0>
      using Layout = portablecollection::TypeResolver<Idx, T0, Args...>;

      template <std::size_t Idx = 0UL>
      using View = typename std::tuple_element<Idx, std::tuple<T0, Args...>>::type::View;

      template <std::size_t Idx = 0UL>
      using ConstView = typename  std::tuple_element<Idx, std::tuple<T0, Args...>>::type::ConstView;

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
      Leaf<index_t_ <T>>& get() {
          return static_cast<Leaf<index_t_<T>>&>(impl_);
      }

      template <typename T>
      Leaf<index_t_<T>> const& get() const {
          return static_cast<Leaf<index_t_<T>> const&>(impl_);
      }

 public:
  PortableDeviceMultiCollection() = default;

  PortableDeviceMultiCollection(int32_t elements, cudaStream_t stream)
      : buffer_{cms::cuda::make_device_unique<std::byte[]>(Layout<>::computeDataSize(elements), stream)},
    impl_{buffer_.get(), elements} {
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  static int32_t computeDataSize(const SizesArray& sizes) {
    int32_t ret = 0;
    portablecollection::constexpr_for<0, members_>(
        [&sizes, &ret](auto i) { ret += Layout<i>::computeDataSize(sizes[i]); });
    return ret;
  }

  PortableDeviceMultiCollection(const SizesArray& sizes, cudaStream_t stream)
      // allocate device memory asynchronously on the given work queue
      : buffer_{cms::cuda::make_device_unique<std::byte[]>(computeDataSize(sizes), stream)},
        impl_{buffer_.get(), sizes} {
    portablecollection::constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    portablecollection::constexpr_for<1, members_>(
        [&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  // non-copyable
  PortableDeviceMultiCollection(PortableDeviceMultiCollection const&) = delete;
  PortableDeviceMultiCollection& operator=(PortableDeviceMultiCollection const&) = delete;

  // movable
  PortableDeviceMultiCollection(PortableDeviceMultiCollection&&) = default;
  PortableDeviceMultiCollection& operator=(PortableDeviceMultiCollection&&) = default;

  // default destructor
  ~PortableDeviceMultiCollection() = default;

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

  size_t bufferSize() const {
      SizesArray layoutSize;
      size_t bytes;
      bytes = 0;
      portablecollection::constexpr_for<0, members_>([&](auto i) {
              layoutSize[i] = get<i>().layout_.metadata().byteSize();
              bytes += layoutSize[i];
              });
      return bytes;
  }


private:
  Buffer buffer_;  //!
  Implementation impl_;           // (serialized: this is where the layouts live)
};

// Singleton case does not need to be aliased. A special template covers it.

  // This aliasing is needed to work with ROOT serialization. Bare templates make dictionary compilation fail.
  template <typename T0, typename T1>
  using PortableDeviceCollection2 = PortableDeviceMultiCollection<T0, T1>;

  template <typename T0, typename T1, typename T2>
  using PortableDeviceCollection3 = PortableDeviceMultiCollection<T0, T1, T2>;

  template <typename T0, typename T1, typename T2, typename T3>
  using PortableDeviceCollection4 = PortableDeviceMultiCollection<T0, T1, T2, T3>;

  template <typename T0, typename T1, typename T2, typename T3, typename T4>
  using PortableDeviceCollection5 = PortableDeviceMultiCollection<T0, T1, T2, T3, T4>;

}  // namespace cms::cuda

#endif  // CUDADataFormats_Common_interface_PortableDeviceCollection_h
