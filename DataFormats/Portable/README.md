## Define portable data formats that wrap SoA data structures and can be persisted to ROOT files

### `PortableHostCollection<T>`

`PortableHostCollection<T>` is a class template that wraps a SoA type `T` and an alpaka host buffer, which owns the
memory where the SoA is allocated. The content of the SoA is persistent, while the buffer itself is transient.
Specialisations of this template can be persisted, and can be read back also in "bare ROOT" mode, without any
dictionaries.
They have no implicit or explicit references to alpaka (neither as part of the class signature nor as part of its name).
This could make it possible to read them back with different portability solutions in the future.

### `PortableDeviceCollection<T, TDev>`

`PortableDeviceCollection<T, TDev>` is a class template that wraps a SoA type `T` and an alpaka device buffer, which
owns the memory where the SoA is allocated.
To avoid confusion and ODR-violations, the `PortableDeviceCollection<T, TDev>` template cannot be used with the `Host`
device type.
Specialisations of this template are transient and cannot be persisted.

### `ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>`

`ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>` is a template alias that resolves to either
`PortableHostCollection<T>` or `PortableDeviceCollection<T, ALPAKA_ACCELERATOR_NAMESPACE::Device>`, depending on the
backend.

### `PortableCollection<T, TDev>`

`PortableCollection<T, TDev>` is an alias template that resolves to `ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>`
for the matching device.


## Notes

Modules that are supposed to work with only host types (_e.g._ dealing with de/serialisation, data transfers, _etc._)
should explicitly use the `PortableHostCollection<T>` types.

Modules that implement portable interfaces (_e.g._ producers) should use the generic types based on
`ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>` or `PortableCollection<T, TDev>`.

## Multi layout collections

Some use cases require multiple sets of columns of different sizes. This is can be achieved in a single
`PortableCollection` using `PortableCollection2<T1, T2>`, `PortableCollection3<T1, T2, T3>` and so on up to
`PortableCollection5<...>`. The numbered, fixed size wrappers are needed in order to be added to the ROOT dictionary.
Behind the scenes recursive `PortableHostMultiCollection<T0, ...>` and
`ALPAKA_ACCELERATOR_NAMESPACE::PortableDeviceMultiCollection<TDev, T0, ...>` (note the reversed parameter order) provide
the actual class definitions.

## ROOT dictionary declaration helper scripts

In order to be serialized by ROOT, the products need to be added to its dictionary. This happens during `scram build`
as instructed in `<module>/src/classes_dev.xml` and `<module>/src/alpaka/classes_cuda_def.xml` and
`<module>/src/alpaka/classes_rocm_def.xml`. Two scripts generate the code to be added to the xml files.
Both scripts expect the collections to be aliased as in:
```
using TestDeviceMultiCollection3 = PortableCollection3<TestSoA, TestSoA2, TestSoA3>;
```

For the host xml, SoA layouts have to be listed and duplicates should be removed manually is multiple
collections share a same layout. The scripts are called as follows:
```
./DataFormats/Portable/scripts/portableHostCollectionHints portabletest::TestHostMultiCollection3  \
            portabletest::TestSoALayout portabletest::TestSoALayout2 portabletest::TestSoALayout3

./DataFormats/Portable/scripts/portableDeviceCollectionHints portabletest::TestHostMultiCollection3
```
The layouts should not be added as parameters for the device collection. Those script can be use equally with the
single layout collections or multi layout collections.