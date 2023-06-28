#ifndef DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
#define DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h

#include <optional>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // ... or any other CPU-based accelerators

  // generic SoA-based product in host memory
  template <typename T>
  using PortableCollection = ::PortableHostCollection<T>;

#else

  // generic SoA-based product in device memory
  template <typename T>
  using PortableCollection = ::PortableDeviceCollection<T, Device>;

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace traits {

  // specialise the trait for the device provided by the ALPAKA_ACCELERATOR_NAMESPACE
  template <typename T>
  class PortableCollectionTrait<T, ALPAKA_ACCELERATOR_NAMESPACE::Device> {
    using CollectionType = ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>;
  };

}  // namespace traits

namespace cms::alpakatools {
  template <typename TLayout, typename TDevice>
  struct CopyToHost<PortableDeviceCollection<TLayout, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceCollection<TLayout, TDevice> const& srcData) {
      PortableHostCollection<TLayout> dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename TLayout>
  struct CopyToDevice<PortableHostCollection<TLayout>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableHostCollection<TLayout> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceCollection<TLayout, TDevice> dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // Singleton case does not need to be aliased. A special template covers it.

  // This aliasing is needed to work with ROOT serialization. Bare templates make dictionary compilation fail.
  template <typename T0, typename T1>
  using PortableCollection2 = ::PortableHostMultiCollection<T0, T1>;

  template <typename T0, typename T1, typename T2>
  using PortableCollection3 = ::PortableHostMultiCollection<T0, T1, T2>;

  template <typename T0, typename T1, typename T2, typename T3>
  using PortableCollection4 = ::PortableHostMultiCollection<T0, T1, T2, T3>;

  template <typename T0, typename T1, typename T2, typename T3, typename T4>
  using PortableCollection5 = ::PortableHostMultiCollection<T0, T1, T2, T3, T4>;
#else
  // Singleton case does not need to be aliased. A special template covers it.

  // This aliasing is needed to work with ROOT serialization. Bare templates make dictionary compilation fail.
  template <typename T0, typename T1>
  using PortableCollection2 = ::PortableDeviceMultiCollection<Device, T0, T1>;

  template <typename T0, typename T1, typename T2>
  using PortableCollection3 = ::PortableDeviceMultiCollection<Device, T0, T1, T2>;

  template <typename T0, typename T1, typename T2, typename T3>
  using PortableCollection4 = ::PortableDeviceMultiCollection<Device, T0, T1, T2, T3>;

  template <typename T0, typename T1, typename T2, typename T3, typename T4>
  using PortableCollection5 = ::PortableDeviceMultiCollection<Device, T0, T1, T2, T3, T4>;
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace traits {
// specialise the trait for the device provided by the ALPAKA_ACCELERATOR_NAMESPACE
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <typename T0, typename... Args>
  class PortableMultiCollectionTrait<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, Args...> {
    using CollectionType = ::PortableHostMultiCollection<T0, Args...>;
  };
#else
  template <typename T0, typename... Args>
  class PortableMultiCollectionTrait<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, Args...> {
    using CollectionType = ::PortableDeviceMultiCollection<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, Args...>;
  };
#endif

}  // namespace traits

namespace cms::alpakatools {
  template <typename TDevice, typename T0, typename... Args>
  struct CopyToHost<PortableDeviceMultiCollection<TDevice, T0, Args...>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceMultiCollection<TDevice, T0, Args...> const& srcData) {
      PortableHostMultiCollection<T0, Args...> dstData(srcData.sizes(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename T0, typename... Args>
  struct CopyToDevice<PortableHostMultiCollection<T0, Args...>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableHostMultiCollection<T0, Args...> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceMultiCollection<TDevice, T0, Args...> dstData(srcData.sizes(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
